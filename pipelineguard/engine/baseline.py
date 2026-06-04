"""Rolling baseline manager per stage per metric."""

from collections import defaultdict, deque

import numpy as np


class BaselineManager:
    """Maintains sliding-window statistics for each (stage, metric) pair."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        # stage -> metric -> deque of float values
        self._baselines: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window_size))
        )

    def update(self, stage: str, metrics: dict) -> None:
        """Add current metric values to the rolling window."""
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._baselines[stage][metric].append(float(value))

    def get_stats(self, stage: str, metric: str) -> dict | None:
        """Return mean/std/n for a metric, or None if insufficient data."""
        values = self._baselines.get(stage, {}).get(metric)
        if values is None or len(values) < 5:
            return None
        arr = np.array(list(values), dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }

    def has_baseline(self, stage: str, metric: str) -> bool:
        """True if we have at least 10 readings for this metric."""
        values = self._baselines.get(stage, {}).get(metric)
        return values is not None and len(values) >= 10

    def snapshot(self) -> dict:
        """Return a serializable snapshot of all current baselines."""
        snap = {}
        for stage, metrics in self._baselines.items():
            snap[stage] = {}
            for metric, vals in metrics.items():
                snap[stage][metric] = list(vals)
        return snap
