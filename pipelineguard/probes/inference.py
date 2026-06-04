"""Probe for the Inference pipeline stage."""

from datetime import datetime, timezone

import numpy as np

from probes.base import BaseProbe, ProbeReading


class InferenceProbe(BaseProbe):
    """Measures model confidence, risk distribution, and latency."""

    def __init__(self, telemetry_store):
        super().__init__("inference", telemetry_store)

    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        n = len(records)
        risk_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0, "unknown": 0}
        confidences = []

        for rec in records:
            ir = rec.get("inference_result", {})
            risk = ir.get("risk_level", "unknown")
            conf = ir.get("confidence", 0.0)
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
            confidences.append(conf)

        avg_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0
        std_conf = round(float(np.std(confidences)), 3) if confidences else 0.0
        unknown_rate = round(risk_dist.get("unknown", 0) / n, 4) if n > 0 else 0.0

        metrics = {
            "record_count": n,
            "avg_confidence": avg_conf,
            "confidence_std": std_conf,
            "unknown_risk_rate": unknown_rate,
            "inference_time_ms": stage_metadata.get("inference_time_ms", 0),
            **{f"risk_{k}": v for k, v in risk_dist.items()},
        }

        return ProbeReading(
            stage=self.stage_name,
            timestamp=datetime.now(timezone.utc),
            batch_id=stage_metadata.get("batch_id", ""),
            record_count=n,
            metrics=metrics,
            metadata=stage_metadata,
            raw_sample=records[:3],
        )
