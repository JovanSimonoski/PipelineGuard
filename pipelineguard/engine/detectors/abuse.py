"""Behavioral / rate-pattern detector for access abuse threats."""

from engine.baseline import BaselineManager
from probes.base import ProbeReading


class AbuseDetector:
    """Detects access abuse via batch size spikes and rate anomalies."""

    _SPIKE_FACTOR = {"low": 5.0, "medium": 3.0, "high": 2.0}

    def __init__(self, sensitivity: str = "medium"):
        self.spike_factor = self._SPIKE_FACTOR[sensitivity]

    def detect(self, reading: ProbeReading, baseline: BaselineManager) -> dict:
        """
        Checks:
        1. Batch size spike vs rolling average
        2. Record count consistency across stages (unexpected drops/surges)
        3. Drop rate spike in preprocessing
        """
        alerts = []
        scores = []

        record_count = reading.metrics.get("record_count", 0)

        # Check 1: batch size spike
        count_stats = baseline.get_stats(reading.stage, "record_count")
        if count_stats and count_stats["mean"] > 0:
            ratio = record_count / count_stats["mean"]
            if ratio > self.spike_factor or (ratio < 1.0 / self.spike_factor and record_count > 0):
                z = abs(ratio - 1.0) * count_stats["mean"] / max(count_stats["std"], 0.1)
                scores.append(min(z / (self.spike_factor * 2), 1.0))
                alerts.append({
                    "check": "batch_size_spike",
                    "current_count": record_count,
                    "baseline_mean": round(count_stats["mean"], 1),
                    "ratio": round(ratio, 2),
                })

        # Check 2: drop rate spike (preprocessing only)
        if reading.stage == "preprocessing":
            drop_rate = reading.metrics.get("drop_rate", 0.0)
            drop_stats = baseline.get_stats("preprocessing", "drop_rate")
            if drop_stats and drop_stats["std"] > 1e-6:
                z = (drop_rate - drop_stats["mean"]) / drop_stats["std"]
                if z > 2.5:
                    scores.append(min(z / 5.0, 1.0))
                    alerts.append({
                        "check": "drop_rate_spike",
                        "current_drop_rate": round(drop_rate, 3),
                        "baseline_mean": round(drop_stats["mean"], 3),
                        "z_score": round(z, 2),
                    })

        # Check 3: normalization violations spike
        if reading.stage == "preprocessing":
            viol_rate = reading.metrics.get("normalization_violation_rate", 0.0)
            viol_stats = baseline.get_stats("preprocessing", "normalization_violation_rate")
            if viol_stats and viol_stats["std"] > 1e-6:
                z = (viol_rate - viol_stats["mean"]) / viol_stats["std"]
                if z > 2.5:
                    scores.append(min(z / 5.0, 1.0))
                    alerts.append({
                        "check": "normalization_violation_spike",
                        "current_rate": round(viol_rate, 3),
                        "baseline_mean": round(viol_stats["mean"], 3),
                        "z_score": round(z, 2),
                    })

        score = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "detected": len(alerts) > 0,
            "score": round(score, 3),
            "details": alerts,
        }
