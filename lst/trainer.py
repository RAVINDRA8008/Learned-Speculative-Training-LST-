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
from lst.utils import WeightSnapshot, MetricsTracker


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
        )

        # Weight snapshot for rollback
        self.snapshot = WeightSnapshot()

        # Metrics
        self.metrics = MetricsTracker()
        self.step = 0
        self.rank = rank

    def step_batch(self, batch: dict, lr: float = None) -> dict:
        """
        Execute one LST training step.

        Args:
            batch: Input batch dict (input_ids, attention_mask, labels).
            lr:    Current learning rate (for feature extraction).

        Returns:
            Dict with 'loss', 'accepted' (bool or None), 'acceptance_rate', etc.
        """
        self.step += 1
        if lr is None:
            lr = self.optimizer.defaults.get("lr", 1e-4)

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # === Phase 1: Warmup — standard training, collect gradient data ===
        if self.step <= self.warmup_steps:
            result = self._standard_step(batch, train_draft=True)
            result["phase"] = "warmup"
            result["accepted"] = None
            self._log_step(result, lr)
            return result

        # === Forced supervision every K steps ===
        if self.step % self.K == 0:
            result = self._standard_step(batch, train_draft=True)
            result["phase"] = "forced_supervision"
            result["accepted"] = None
            self._log_step(result, lr)
            return result

        # === Phase 2: Speculative training ===
        return self._speculative_step(batch, lr)

    def _speculative_step(self, batch: dict, lr: float) -> dict:
        """Execute a speculative training step: predict → apply → verify → accept/reject."""

        # Use cached loss from previous step (avoids redundant forward pass)
        current_loss = self._cached_loss if self._cached_loss is not None else 0.0

        # Update baseline if not set
        if self.verifier.baseline_loss is None:
            self.verifier.update_baseline(current_loss)

        # Extract features for draft model
        features = self.feat_extractor.extract(current_loss, self.step, lr)

        # Draft prediction
        self.draft.eval()
        with torch.no_grad():
            predicted_updates = self.draft(features)

        # Snapshot current weights for potential rollback
        self.snapshot.save(self.target_layers)

        # Apply speculative update
        self._apply_speculative_update(predicted_updates, lr)

        # Verify via forward pass (only costly operation per accepted step)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                verify_output = self.model(**batch)
                verify_loss = verify_output.loss.item()

        # Accept or reject
        accepted = self.verifier.should_accept(verify_loss)

        if accepted:
            # Keep the speculative update — skip backward pass entirely
            self._cached_loss = verify_loss
            result = {
                "loss": verify_loss,
                "accepted": True,
                "phase": "speculative",
                "draft_loss": None,
            }
        else:
            # Reject: restore weights and do real training
            self.snapshot.restore(self.target_layers)
            result = self._standard_step(batch, train_draft=True)
            result["accepted"] = False
            result["phase"] = "speculative_rejected"

        result["acceptance_rate"] = self.verifier.acceptance_rate
        result["recent_acceptance_rate"] = self.verifier.recent_acceptance_rate
        result["tolerance"] = self.verifier.tolerance
        result["baseline_loss"] = self.verifier.baseline_loss

        self._log_step(result, lr)
        return result

    def _standard_step(self, batch: dict, train_draft: bool = False) -> dict:
        """Execute a standard training step with backward pass."""
        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            output = self.model(**batch)
            loss = output.loss
        loss.backward()

        # Record gradients before optimizer step
        self.feat_extractor.record_gradients()

        # Train draft model on real gradients
        draft_loss_val = None
        if train_draft and self.step > 10:
            draft_loss_val = self._train_draft(loss.item())

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        self._cached_loss = loss.item()

        return {
            "loss": loss.item(),
            "draft_loss": draft_loss_val,
        }

    def _train_draft(self, loss_val: float) -> float:
        """Train the draft model to predict real gradients (self-supervision)."""
        self.draft.train()

        # Get current LR from optimizer
        lr = self.optimizer.param_groups[0]["lr"]

        # Extract features (same as what draft would see)
        features = self.feat_extractor.extract(loss_val, self.step, lr)

        # Draft prediction
        predicted_updates = self.draft(features)

        # Compare against real gradients (projected to low-rank space)
        draft_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for i, (name, param) in enumerate(self.target_layers):
            if param.grad is None:
                continue

            # Get the real gradient and project to low-rank space
            real_grad = param.grad.detach()
            pred = predicted_updates[i]

            # Decode the prediction using the layer-specific decoder
            decoded_pred = self.draft.decode_update(pred, param.shape, layer_idx=i)
            target_grad_normalized = real_grad / (real_grad.norm() + 1e-8)
            decoded_pred_normalized = decoded_pred / (decoded_pred.norm() + 1e-8)

            # Loss: MSE on normalized gradients + direction loss
            mse_loss = (decoded_pred - real_grad).pow(2).mean()
            cosine_loss = 1.0 - (target_grad_normalized * decoded_pred_normalized).sum()

            draft_loss = draft_loss + mse_loss + 0.1 * cosine_loss
            n_valid += 1

        if n_valid > 0:
            draft_loss = draft_loss / n_valid
            self.draft_optimizer.zero_grad()
            draft_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.draft.parameters(), 1.0)
            self.draft_optimizer.step()
            return draft_loss.item()

        return None

    def _apply_speculative_update(self, updates: list, lr: float):
        """Apply the draft model's predicted updates to target model weights."""
        with torch.no_grad():
            for i, (name, param) in enumerate(self.target_layers):
                update = self.draft.decode_update(updates[i], param.shape, layer_idx=i)
                # Apply as a gradient step: w = w - lr * predicted_gradient
                param.data.add_(update, alpha=-lr)

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
