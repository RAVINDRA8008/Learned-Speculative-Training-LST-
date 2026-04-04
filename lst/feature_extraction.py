"""
Feature extraction for LST: extracts per-layer features from the target model
including weight statistics, compressed gradient history, loss, and training progress.

Uses random projection (Johnson-Lindenstrauss) to compress high-dimensional
gradients into fixed-size vectors while preserving cosine similarity structure.
"""

import torch
from collections import deque


class GradientHistoryBuffer:
    """
    Stores compressed gradient history for each tracked parameter layer.
    Gradients are compressed via a fixed random projection matrix.
    """

    def __init__(self, proj_dim: int = 32, history_len: int = 4, device: str = "cpu"):
        self.proj_dim = proj_dim
        self.history_len = history_len
        self.device = device
        self._history = {}        # layer_idx -> deque of projected gradients
        self._proj_matrices = {}  # layer_idx -> random projection matrix
        self._grad_norms = {}     # layer_idx -> deque of gradient norms

    def _get_proj_matrix(self, layer_idx: int, grad_numel: int) -> torch.Tensor:
        """Get or create a fixed random projection matrix for a layer."""
        if layer_idx not in self._proj_matrices:
            # Gaussian random projection (JL-lemma guarantee)
            # Generate on CPU for determinism, then move to device
            gen = torch.Generator(device='cpu')
            gen.manual_seed(42 + layer_idx)
            mat = torch.randn(grad_numel, self.proj_dim, generator=gen).to(self.device)
            mat = mat / (grad_numel ** 0.5)  # normalize for variance preservation
            self._proj_matrices[layer_idx] = mat
            self._history[layer_idx] = deque(maxlen=self.history_len)
            self._grad_norms[layer_idx] = deque(maxlen=self.history_len)
        return self._proj_matrices[layer_idx]

    def push(self, layer_idx: int, gradient: torch.Tensor):
        """Store a compressed version of the gradient."""
        flat = gradient.detach().flatten().to(self.device)
        proj_mat = self._get_proj_matrix(layer_idx, flat.numel())
        projected = flat @ proj_mat  # (proj_dim,)
        self._history[layer_idx].append(projected)
        self._grad_norms[layer_idx].append(flat.norm().item())

    def get_features(self, layer_idx: int) -> torch.Tensor:
        """
        Get the full gradient history feature vector for a layer.
        Returns: (history_len * proj_dim,) tensor of projected gradient history.
        """
        if layer_idx not in self._history or len(self._history[layer_idx]) == 0:
            return torch.zeros(self.history_len * self.proj_dim, device=self.device)

        hist = list(self._history[layer_idx])
        # Pad with zeros if we don't have enough history yet
        while len(hist) < self.history_len:
            hist.insert(0, torch.zeros(self.proj_dim, device=self.device))

        return torch.cat(hist[-self.history_len:])

    def get_norms(self, layer_idx: int) -> list:
        """Get recent gradient norms for a layer."""
        if layer_idx not in self._grad_norms:
            return [0.0] * self.history_len
        norms = list(self._grad_norms[layer_idx])
        while len(norms) < self.history_len:
            norms.insert(0, 0.0)
        return norms[-self.history_len:]

    def to(self, device):
        """Move all buffers to a device."""
        self.device = device
        for idx in self._proj_matrices:
            self._proj_matrices[idx] = self._proj_matrices[idx].to(device)
        for idx in self._history:
            self._history[idx] = deque(
                [h.to(device) for h in self._history[idx]],
                maxlen=self.history_len,
            )
        return self


class FeatureExtractor:
    """
    Extracts per-layer feature vectors from the target model for input
    to the GradientTransformer draft model.

    Per-layer features (134 floats):
      - Weight statistics: mean, std, norm (3 floats)
      - Gradient history: last 4 gradient projections (4 × 32 = 128 floats)
      - Current batch loss (1 float)
      - Training progress: step / total_steps (1 float)
      - Current learning rate (1 float)

    Total: 3 + 128 + 1 + 1 + 1 = 134 floats per layer
    """

    def __init__(
        self,
        target_layers: list,
        proj_dim: int = 32,
        history_len: int = 4,
        total_steps: int = 100_000,
        device: str = "cpu",
    ):
        """
        Args:
            target_layers: List of (name, param) tuples for tracked parameters.
            proj_dim:      Dimensionality of random projection for gradient compression.
            history_len:   Number of past gradients to keep in history.
            total_steps:   Total training steps (for progress normalization).
            device:        Device for feature tensors.
        """
        self.target_layers = target_layers
        self.proj_dim = proj_dim
        self.history_len = history_len
        self.total_steps = total_steps
        self.device = device
        self.feat_dim = 3 + (history_len * proj_dim) + 3  # weight_stats + grad_hist + global

        self.grad_buffer = GradientHistoryBuffer(
            proj_dim=proj_dim, history_len=history_len, device=device
        )

    def record_gradients(self):
        """Record current gradients from all target layers into the history buffer."""
        for i, (name, param) in enumerate(self.target_layers):
            if param.grad is not None:
                self.grad_buffer.push(i, param.grad)

    def extract(self, loss_val: float, step: int, lr: float) -> torch.Tensor:
        """
        Extract features from all target layers.

        Args:
            loss_val: Current batch loss value.
            step:     Current training step.
            lr:       Current learning rate.

        Returns:
            (n_layers, feat_dim) tensor of features.
        """
        features = []
        for i, (name, param) in enumerate(self.target_layers):
            w = param.data

            # Weight statistics (3 floats)
            weight_stats = torch.tensor(
                [w.mean().item(), w.std().item(), w.norm().item()],
                device=self.device,
            )

            # Gradient history (history_len * proj_dim floats)
            grad_hist = self.grad_buffer.get_features(i)

            # Global info (3 floats)
            global_info = torch.tensor(
                [loss_val, step / max(self.total_steps, 1), lr],
                device=self.device,
            )

            layer_feat = torch.cat([weight_stats, grad_hist, global_info])
            features.append(layer_feat)

        return torch.stack(features)  # (n_layers, feat_dim)

    def to(self, device):
        self.device = device
        self.grad_buffer.to(device)
        return self
