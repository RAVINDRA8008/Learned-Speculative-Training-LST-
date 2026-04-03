"""
Utility functions for LST: weight snapshots, restore, metrics tracking.
"""

import torch
import copy
from collections import defaultdict


class WeightSnapshot:
    """Efficient weight snapshot and restore for speculative rollback."""

    def __init__(self):
        self._snapshot = {}

    def save(self, named_params):
        """Save a snapshot of current parameter values."""
        self._snapshot = {
            name: param.data.clone() for name, param in named_params
        }

    def restore(self, named_params):
        """Restore parameters from the last snapshot."""
        for name, param in named_params:
            if name in self._snapshot:
                param.data.copy_(self._snapshot[name])

    def clear(self):
        self._snapshot = {}


class MetricsTracker:
    """Tracks training metrics over time for logging and plotting."""

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, step: int, **kwargs):
        self.history["step"].append(step)
        for k, v in kwargs.items():
            self.history[k].append(v)

    def get(self, key: str) -> list:
        return self.history.get(key, [])

    def get_recent(self, key: str, n: int = 100):
        vals = self.history.get(key, [])
        return vals[-n:]

    def summary(self, last_n: int = 100) -> dict:
        result = {}
        for key, vals in self.history.items():
            if key == "step":
                continue
            recent = vals[-last_n:]
            if recent and isinstance(recent[0], (int, float)):
                result[f"{key}_mean"] = sum(recent) / len(recent)
                result[f"{key}_last"] = recent[-1]
        return result
