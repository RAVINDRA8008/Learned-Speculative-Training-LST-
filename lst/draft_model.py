"""
GradientTransformer: A 3M-parameter draft model that predicts weight updates
for the target model by learning its gradient manifold.

Architecture:
  Input (n_layers × feat_dim) → Linear → +PosEmbed → 4× TransformerBlock → Per-layer heads → Low-rank ΔW
"""

import torch
import torch.nn as nn


class LayerDecoder(nn.Module):
    """Decodes compact code vector into full weight update using learned basis vectors."""

    def __init__(self, d_out, d_in, rank):
        super().__init__()
        self.rank = rank
        self.d_out = d_out
        self.d_in = d_in
        self.A_basis = nn.Parameter(torch.randn(d_out, rank) * 0.01)
        self.B_basis = nn.Parameter(torch.randn(d_in, rank) * 0.01)

    def forward(self, code):
        """
        Args:
            code: (2*rank,) scaling coefficients from draft head
        Returns:
            (d_out, d_in) weight update
        """
        scale_a = code[:self.rank]
        scale_b = code[self.rank:]
        A = self.A_basis * scale_a.unsqueeze(0)  # (d_out, rank)
        B = self.B_basis * scale_b.unsqueeze(0)  # (d_in, rank)
        return A @ B.T  # (d_out, d_in)


class GradientTransformer(nn.Module):
    """
    Draft model that predicts low-rank weight updates for each layer
    of the target model, given per-layer feature vectors summarizing
    weight statistics, gradient history, loss, and training progress.
    """

    def __init__(
        self,
        n_layers: int = 12,
        feat_dim: int = 134,
        d_model: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        rank: int = 32,
        layer_dims: list = None,
    ):
        """
        Args:
            n_layers:   Number of target model parameter groups to predict updates for.
            feat_dim:   Dimensionality of per-layer input features.
            d_model:    Hidden dimension of transformer.
            n_heads:    Number of attention heads.
            n_blocks:   Number of transformer encoder layers.
            rank:       Rank of the low-rank weight update approximation.
            layer_dims: List of (d_out, d_in) tuples for each target layer.
                        Used to size per-layer output heads. If None, heads
                        output a generic (rank * 2) vector.
        """
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_model = d_model
        self.layer_dims = layer_dims

        # Input projection
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Embedding(n_layers, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Compact per-layer heads: output 2*rank coefficients (NOT full factors)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 2 * rank)
            for _ in range(n_layers)
        ])

        # Per-layer decoders: learned basis vectors for efficient low-rank updates
        if layer_dims is not None:
            self.decoders = nn.ModuleList([
                LayerDecoder(d_out, d_in, rank)
                for d_out, d_in in layer_dims
            ])
        else:
            self.decoders = None

        self._init_weights()

    def _init_weights(self):
        """Initialize output heads with small weights for stable early predictions."""
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, layer_features: torch.Tensor):
        """
        Args:
            layer_features: (n_layers, feat_dim) tensor of per-layer features.

        Returns:
            List of n_layers tensors, each containing the raw low-rank factors
            for that layer's predicted weight update.
        """
        # Project input features to model dimension
        x = self.input_proj(layer_features)  # (n_layers, d_model)

        # Add positional encoding (layer index)
        pos_ids = torch.arange(self.n_layers, device=x.device)
        x = x + self.pos_embed(pos_ids)

        # Transformer expects (batch, seq, dim)
        x = x.unsqueeze(0)  # (1, n_layers, d_model)
        x = self.transformer(x)
        x = x.squeeze(0)  # (n_layers, d_model)

        # Per-layer heads produce low-rank factor vectors
        updates = []
        for i, head in enumerate(self.heads):
            factors = head(x[i])  # (output_dim,)
            updates.append(factors)

        return updates

    def decode_update(self, code: torch.Tensor, shape: tuple, layer_idx: int = None) -> torch.Tensor:
        """
        Decode compact code into a full weight update tensor via learned basis.

        Args:
            code:      Compact code from a per-layer head (2*rank values).
            shape:     Shape of the target parameter.
            layer_idx: Index into self.decoders for basis-based decoding.

        Returns:
            Weight update tensor matching `shape`.
        """
        if self.decoders is not None and layer_idx is not None:
            return self.decoders[layer_idx](code)

        # Fallback: simple outer-product approach
        if len(shape) == 2:
            d_out, d_in = shape
            half = min(self.rank, code.numel() // 2)
            a_scale = code[:half]
            b_scale = code[half:2 * half]
            A = a_scale.unsqueeze(0).expand(d_out, -1) * 0.01
            B = b_scale.unsqueeze(0).expand(d_in, -1) * 0.01
            return A @ B.T
        elif len(shape) == 1:
            return code[:shape[0]]
        else:
            numel = 1
            for s in shape:
                numel *= s
            flat = code[:numel] if code.numel() >= numel else torch.cat([
                code, torch.zeros(numel - code.numel(), device=code.device)
            ])
            return flat.view(shape)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
