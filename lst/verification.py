"""
Verification module for LST: accept/reject logic and adaptive tolerance scheduling.
"""

import torch
from collections import deque


class Verifier:
    """
    Handles the accept/reject decision for speculative weight updates
    and adapts the tolerance threshold based on recent acceptance rates.
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        ema_decay: float = 0.99,
        adaptive: bool = True,
        tol_min: float = 0.005,
        tol_max: float = 0.05,
        target_accept_low: float = 0.45,
        target_accept_high: float = 0.85,
        window_size: int = 100,
    ):
        self.tolerance = tolerance
        self.ema_decay = ema_decay
        self.adaptive = adaptive
        self.tol_min = tol_min
        self.tol_max = tol_max
        self.target_accept_low = target_accept_low
        self.target_accept_high = target_accept_high

        self.baseline_loss = None
        self.recent_decisions = deque(maxlen=window_size)

        # Tracking
        self.total_accepted = 0
        self.total_speculative = 0

    def update_baseline(self, loss_val: float):
        """Update the EMA baseline loss."""
        if self.baseline_loss is None:
            self.baseline_loss = loss_val
        else:
            self.baseline_loss = (
                self.ema_decay * self.baseline_loss
                + (1 - self.ema_decay) * loss_val
            )

    def should_accept(self, verify_loss: float) -> bool:
        """
        Determine whether to accept the speculative update.

        Args:
            verify_loss: Loss computed on the speculatively-updated model.

        Returns:
            True if accepted, False if rejected.
        """
        if self.baseline_loss is None:
            return False

        threshold = self.baseline_loss * (1.0 + self.tolerance)
        accepted = verify_loss < threshold

        self.total_speculative += 1
        self.recent_decisions.append(1 if accepted else 0)

        if accepted:
            self.total_accepted += 1
            # NOTE: Do NOT update baseline from speculative accepts.
            # Baseline is only updated from real gradient steps (warmup/supervision)
            # to prevent cumulative drift.

        # Adapt tolerance after accumulating enough decisions
        if self.adaptive and len(self.recent_decisions) >= 20:
            self._adapt_tolerance()

        return accepted

    def _adapt_tolerance(self):
        """Adjust tolerance based on recent acceptance rate."""
        recent_rate = sum(self.recent_decisions) / len(self.recent_decisions)

        if recent_rate > self.target_accept_high:
            # Too lenient — tighten
            self.tolerance *= 0.95
        elif recent_rate < self.target_accept_low:
            # Too strict — loosen
            self.tolerance *= 1.05

        # Clamp
        self.tolerance = max(self.tol_min, min(self.tol_max, self.tolerance))

    @property
    def acceptance_rate(self) -> float:
        if self.total_speculative == 0:
            return 0.0
        return self.total_accepted / self.total_speculative

    @property
    def recent_acceptance_rate(self) -> float:
        if len(self.recent_decisions) == 0:
            return 0.0
        return sum(self.recent_decisions) / len(self.recent_decisions)

    def get_stats(self) -> dict:
        return {
            "acceptance_rate": self.acceptance_rate,
            "recent_acceptance_rate": self.recent_acceptance_rate,
            "tolerance": self.tolerance,
            "baseline_loss": self.baseline_loss,
            "total_accepted": self.total_accepted,
            "total_speculative": self.total_speculative,
        }
