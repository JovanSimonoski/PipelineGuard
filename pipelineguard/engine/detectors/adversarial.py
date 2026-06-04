"""Out-of-distribution and confidence-collapse detector for adversarial inputs."""

from engine.baseline import BaselineManager
from probes.base import ProbeReading


class AdversarialDetector:
    """
    Detects adversarial inputs by monitoring model confidence and OOD patterns.
    """

    _LOW_CONF = {"low": 0.20, "medium": 0.35, "high": 0.50}
    _UNKNOWN_RATE = {"low": 0.30, "medium": 0.15, "high": 0.05}
    _CONF_STD_MULT = {"low": 4.0, "medium": 2.5, "high": 1.5}

    def __init__(self, sensitivity: str = "medium"):
        self.low_conf_threshold = self._LOW_CONF[sensitivity]
        self.unknown_rate_threshold = self._UNKNOWN_RATE[sensitivity]
        self.conf_std_multiplier = self._CONF_STD_MULT[sensitivity]

    def detect(self, reading: ProbeReading, baseline: BaselineManager) -> dict:
        """
        Checks:
        1. Avg confidence dropped significantly vs baseline
        2. Rate of 'unknown' risk level is abnormally high
        3. Confidence std dev unusually high (model confused)
        """
        if reading.stage != "inference":
            return {"detected": False, "score": 0.0, "details": []}

        alerts = []
        scores = []

        avg_conf = reading.metrics.get("avg_confidence", 1.0)
        conf_std = reading.metrics.get("confidence_std", 0.0)
        unknown_rate = reading.metrics.get("unknown_risk_rate", 0.0)

        # Check 1: absolute low confidence floor
        if avg_conf < self.low_conf_threshold:
            score = 1.0 - avg_conf
            scores.append(score)
            alerts.append({
                "check": "low_avg_confidence",
                "value": avg_conf,
                "threshold": self.low_conf_threshold,
            })

        # Check 2: confidence drop vs baseline
        stats = baseline.get_stats("inference", "avg_confidence")
        if stats and stats["std"] > 1e-6:
            conf_z = (stats["mean"] - avg_conf) / stats["std"]
            if conf_z > 2.0:
                scores.append(min(conf_z / 3.0, 1.0))
                alerts.append({
                    "check": "confidence_drop_vs_baseline",
                    "current": round(avg_conf, 3),
                    "baseline_mean": round(stats["mean"], 3),
                    "z_score": round(conf_z, 2),
                })

        # Check 3: abnormally high unknown rate
        if unknown_rate > self.unknown_rate_threshold:
            score = min(unknown_rate / self.unknown_rate_threshold, 1.0)
            scores.append(score)
            alerts.append({
                "check": "high_unknown_rate",
                "value": round(unknown_rate, 3),
                "threshold": self.unknown_rate_threshold,
            })

        # Check 4: high confidence variance (confused model)
        std_stats = baseline.get_stats("inference", "confidence_std")
        if std_stats and std_stats["std"] > 1e-6:
            std_z = (conf_std - std_stats["mean"]) / std_stats["std"]
            if std_z > self.conf_std_multiplier:
                scores.append(min(std_z / (self.conf_std_multiplier * 2), 1.0))
                alerts.append({
                    "check": "high_confidence_variance",
                    "confidence_std": round(conf_std, 3),
                    "z_score": round(std_z, 2),
                })

        score = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "detected": len(alerts) > 0,
            "score": round(score, 3),
            "details": alerts,
        }
