"""
GradientTransformer: A 3M-parameter draft model that predicts weight updates
for the target model by learning its gradient manifold.

Architecture:
  Input (n_layers × feat_dim) → Linear → +PosEmbed → 4× TransformerBlock → Per-layer heads → Low-rank ΔW
"""

import torch
import torch.nn as nn


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

        # Per-layer output heads
        # Each head outputs the concatenated low-rank factors A and B
        # for that layer's weight update: ΔW ≈ A @ B^T
        if layer_dims is not None:
            self.heads = nn.ModuleList([
                nn.Linear(d_model, (d_out + d_in) * rank)
                for d_out, d_in in layer_dims
            ])
        else:
            # Fallback: generic output size
            self.heads = nn.ModuleList([
                nn.Linear(d_model, rank * 2)
                for _ in range(n_layers)
            ])

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

    def decode_update(self, factors: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Decode low-rank factors into a full weight update tensor.

        Args:
            factors: Raw output from a per-layer head.
            shape:   Shape of the target parameter (e.g., (768, 768) for a linear layer).

        Returns:
            Weight update tensor matching `shape`.
        """
        if len(shape) == 2:
            d_out, d_in = shape
            rank = self.rank
            # Split factors into A (d_out × rank) and B (d_in × rank)
            a = factors[:d_out * rank].view(d_out, rank)
            b = factors[d_out * rank:(d_out + d_in) * rank].view(d_in, rank)
            update = a @ b.T  # (d_out, d_in)
            return update
        elif len(shape) == 1:
            # Bias or 1D parameter — just take first `shape[0]` values
            return factors[:shape[0]]
        else:
            # Higher-dimensional parameter — flatten, fill, reshape
            numel = 1
            for s in shape:
                numel *= s
            flat = factors[:numel] if factors.numel() >= numel else torch.cat([
                factors, torch.zeros(numel - factors.numel(), device=factors.device)
            ])
            return flat.view(shape)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
