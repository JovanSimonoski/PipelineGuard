"""Statistical drift detector for data poisoning threats."""

import numpy as np

from engine.baseline import BaselineManager
from probes.base import ProbeReading


class PoisoningDetector:
    """Detects statistical drift in incoming data batches using Z-score analysis."""

    _THRESHOLDS = {"low": 3.5, "medium": 2.5, "high": 1.5}

    def __init__(self, sensitivity: str = "medium"):
        self.threshold = self._THRESHOLDS[sensitivity]

    def detect(self, reading: ProbeReading, baseline: BaselineManager) -> dict:
        """
        Returns detected flag, a 0-1 anomaly score, and per-metric details.
        """
        alerts = []
        scores = []

        for metric, value in reading.metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not baseline.has_baseline(reading.stage, metric):
                continue
            stats = baseline.get_stats(reading.stage, metric)
            if stats is None or stats["std"] < 1e-6:
                continue

            z = abs((value - stats["mean"]) / stats["std"])
            scores.append(min(z / self.threshold, 1.0))

            if z > self.threshold:
                alerts.append({
                    "metric": metric,
                    "z_score": round(z, 2),
                    "value": value,
                    "baseline_mean": round(stats["mean"], 3),
                    "baseline_std": round(stats["std"], 3),
                })

        score = float(np.mean(scores)) if scores else 0.0
        return {
            "detected": len(alerts) > 0,
            "score": round(score, 3),
            "details": alerts,
        }
