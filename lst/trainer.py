"""
Core LST Training Loop: Learned Speculative Training.

This implements the full draft-predict → speculative-apply → verify → accept/reject
training loop with online self-supervision of the draft model.
"""

import torch
import torch.nn as nn
from typing import Optional
from lst.draft_model import GradientTransformer
from lst.feature_extraction import FeatureExtractor
from lst.verification import Verifier
from lst.utils import MetricsTracker


class LSTTrainer:
    """
    Learned Speculative Training wrapper.

    Wraps a standard model + optimizer training loop and adds speculative
    weight prediction with a learned draft model. The draft model is trained
    online via self-supervision on the target model's real gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        K: int = 10,
        tolerance: float = 0.01,
        warmup_steps: int = 1000,
        grad_history_len: int = 4,
        proj_dim: int = 32,
        rank: int = 32,
        total_steps: int = 100_000,
        draft_lr: float = 3e-4,
        d_model: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        adaptive_tolerance: bool = True,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        draft_layer_fraction: float = 0.25,
        draft_max_elements: int = 4096,
        draft_train_every: int = 2,
        tol_min: float = 0.005,
        tol_max: float = 0.05,
        hybrid_switch_step: int = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.K = K
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self._cached_loss = None
        self.device = next(model.parameters()).device
        self.draft_layer_fraction = draft_layer_fraction
        self.draft_max_elements = draft_max_elements
        self.draft_train_every = draft_train_every
        self._draft_train_counter = 0
        self.tol_min = tol_min
        self.tol_max = tol_max
        self.hybrid_switch_step = hybrid_switch_step

        # Identify target layers (only 2D weight matrices, exclude embeddings)
        self.target_layers = []
        self.all_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.all_params.append((name, param))
                if param.dim() == 2 and 'wte' not in name and 'wpe' not in name and 'lm_head' not in name:
                    self.target_layers.append((name, param))

        n_layers = len(self.target_layers)
        print(f"[LST] Target model has {n_layers} 2D parameter layers for speculative prediction")
        print(f"[LST] Total trainable params: {sum(p.numel() for _, p in self.all_params):,}")

        # Layer dimensions for the draft model heads
        layer_dims = [(p.shape[0], p.shape[1]) for _, p in self.target_layers]

        # Feature extractor
        self.feat_extractor = FeatureExtractor(
            target_layers=self.target_layers,
            proj_dim=proj_dim,
            history_len=grad_history_len,
            total_steps=total_steps,
            device=self.device,
        )
        feat_dim = self.feat_extractor.feat_dim

        # Draft model
        self.draft = GradientTransformer(
            n_layers=n_layers,
            feat_dim=feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            rank=rank,
            layer_dims=layer_dims,
        ).to(self.device)
        print(f"[LST] Draft model parameters: {self.draft.count_parameters():,}")

        self.draft_optimizer = torch.optim.Adam(self.draft.parameters(), lr=draft_lr)

        # Verifier
        self.verifier = Verifier(
            tolerance=tolerance,
            adaptive=adaptive_tolerance,
            tol_min=tol_min,
            tol_max=tol_max,
        )

        # Metrics
        self.metrics = MetricsTracker()
        self.step = 0
        self.rank = rank

    def step_batch(self, batches, lr: float = None) -> dict:
        """
        Execute one LST training step.

        Args:
            batches: Single batch dict or list of micro-batch dicts for gradient accumulation.
            lr:      Current learning rate (for feature extraction).

        Returns:
            Dict with 'loss', 'accepted' (bool or None), 'acceptance_rate', etc.
        """
        self.step += 1
        if lr is None:
            lr = self.optimizer.defaults.get("lr", 1e-4)

        # Normalize: accept single batch dict or list of micro-batches
        if isinstance(batches, dict):
            batches = [batches]
        # Move all micro-batches to device
        batches = [
            {k: v.to(self.device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            for b in batches
        ]

        # === Phase 1: Warmup — standard training, collect gradient data ===
        if self.step <= self.warmup_steps:
            result = self._standard_step(batches, train_draft=True)
            result["phase"] = "warmup"
            result["accepted"] = None
            self._log_step(result, lr)
            return result

        # === Hybrid switch: pure standard training after switch point ===
        if self.hybrid_switch_step is not None and self.step > self.hybrid_switch_step:
            result = self._standard_step(batches, train_draft=False)
            result["phase"] = "hybrid_standard"
            result["accepted"] = None
            self._log_step(result, lr)
            return result

        # === Forced supervision every K steps ===
        if self.step % self.K == 0:
            result = self._standard_step(batches, train_draft=True)
            result["phase"] = "forced_supervision"
            result["accepted"] = None
            self._log_step(result, lr)
            return result

        # === Phase 2: Speculative training ===
        return self._speculative_step(batches, lr)

    def _speculative_step(self, batches: list, lr: float) -> dict:
        """Execute a speculative training step: predict → apply → verify → accept/reject.

        Optimized hot path:
        - Batched decode via forward_decoded() (one bmm per shape group)
        - No snapshot: rollback by reversing the update
        - model.eval() during verify to skip gradient checkpointing overhead
        - Verify on FIRST micro-batch only (O(1) cost regardless of grad_accum)
        - No .item() CUDA syncs in feature extraction
        """
        current_loss = self._cached_loss if self._cached_loss is not None else 0.0

        if self.verifier.baseline_loss is None:
            self.verifier.update_baseline(current_loss)

        # Extract features for draft model
        features = self.feat_extractor.extract(current_loss, self.step, lr)

        # Draft prediction → batched decoded weight updates
        self.draft.eval()
        with torch.no_grad():
            decoded_updates = self.draft.forward_decoded(features)

        # Apply speculative updates (cached for rollback — no snapshot needed)
        with torch.no_grad():
            for i, (name, param) in enumerate(self.target_layers):
                param.data.add_(decoded_updates[i], alpha=-lr)

        # Verify on FIRST micro-batch — eval mode skips gradient checkpointing
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                verify_output = self.model(**batches[0])
                verify_loss = verify_output.loss.item()
        self.model.train()

        # Accept or reject
        accepted = self.verifier.should_accept(verify_loss)

        if accepted:
            self._cached_loss = verify_loss
            result = {
                "loss": verify_loss,
                "accepted": True,
                "phase": "speculative",
                "draft_loss": None,
            }
        else:
            # Rollback by reversing the update (exact: w + (-lr*u) + (+lr*u) = w)
            with torch.no_grad():
                for i, (name, param) in enumerate(self.target_layers):
                    param.data.add_(decoded_updates[i], alpha=+lr)
            del decoded_updates

            result = self._standard_step(batches, train_draft=True)
            result["accepted"] = False
            result["phase"] = "speculative_rejected"

        result["acceptance_rate"] = self.verifier.acceptance_rate
        result["recent_acceptance_rate"] = self.verifier.recent_acceptance_rate
        result["tolerance"] = self.verifier.tolerance
        result["baseline_loss"] = self.verifier.baseline_loss

        self._log_step(result, lr)
        return result

    def _standard_step(self, batches: list, train_draft: bool = False) -> dict:
        """Execute a standard training step with gradient accumulation over micro-batches."""
        self.optimizer.zero_grad()
        n_micro = len(batches)
        total_loss = 0.0

        for micro_batch in batches:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                output = self.model(**micro_batch)
                scaled_loss = output.loss / n_micro
            scaled_loss.backward()
            total_loss += output.loss.item()

        avg_loss = total_loss / n_micro

        # Record gradients before optimizer step
        self.feat_extractor.record_gradients()

        # Train draft model on real gradients
        draft_loss_val = None
        if train_draft and self.step > 10:
            draft_loss_val = self._train_draft(avg_loss)

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        self._cached_loss = avg_loss

        return {
            "loss": avg_loss,
            "draft_loss": draft_loss_val,
        }

    def _train_draft(self, loss_val: float) -> float:
        """Train the draft model to predict real gradients (self-supervision).

        Optimizations vs. naive version:
        - Only trains on a random subset of layers (draft_layer_fraction)
        - Subsamples elements for MSE instead of materializing full matrices
        - Uses AMP for draft forward/backward
        - Called only every draft_train_every supervision steps
        """
        self._draft_train_counter += 1
        if self._draft_train_counter % self.draft_train_every != 0:
            return None

        self.draft.train()

        lr = self.optimizer.param_groups[0]["lr"]
        features = self.feat_extractor.extract(loss_val, self.step, lr)

        # Draft prediction (under AMP)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            predicted_updates = self.draft(features)

        # Pick a random subset of layers to train on
        n_layers = len(self.target_layers)
        n_sample = max(1, int(n_layers * self.draft_layer_fraction))
        layer_indices = torch.randperm(n_layers, device='cpu')[:n_sample].tolist()

        draft_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for i in layer_indices:
            name, param = self.target_layers[i]
            if param.grad is None:
                continue

            real_grad = param.grad.detach()
            pred = predicted_updates[i]

            # Decode prediction
            decoded_pred = self.draft.decode_update(pred, param.shape, layer_idx=i)

            # Subsample elements for MSE instead of full-matrix comparison
            numel = real_grad.numel()
            if numel > self.draft_max_elements:
                idx = torch.randint(0, numel, (self.draft_max_elements,), device=self.device)
                mse_loss = (decoded_pred.view(-1)[idx] - real_grad.view(-1)[idx]).pow(2).mean()
            else:
                mse_loss = (decoded_pred - real_grad).pow(2).mean()

            draft_loss = draft_loss + mse_loss
            n_valid += 1

        if n_valid > 0:
            draft_loss = draft_loss / n_valid
            self.draft_optimizer.zero_grad()
            draft_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.draft.parameters(), 1.0)
            self.draft_optimizer.step()
            return draft_loss.item()

        return None

    def _log_step(self, result: dict, lr: float):
        """Log metrics for this step."""
        log_data = {
            "loss": result["loss"],
            "lr": lr,
        }
        if result.get("accepted") is not None:
            log_data["accepted"] = 1.0 if result["accepted"] else 0.0
        if result.get("acceptance_rate") is not None:
            log_data["acceptance_rate"] = result["acceptance_rate"]
        if result.get("draft_loss") is not None:
            log_data["draft_loss"] = result["draft_loss"]
        if result.get("tolerance") is not None:
            log_data["tolerance"] = result["tolerance"]
        self.metrics.log(self.step, **log_data)

    def get_stats(self) -> dict:
        """Get comprehensive training statistics."""
        return {
            "step": self.step,
            "verifier": self.verifier.get_stats(),
            "draft_params": self.draft.count_parameters(),
            "metrics_summary": self.metrics.summary(),
        }
