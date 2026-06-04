"""Probe for the Post-processing pipeline stage."""

from datetime import datetime, timezone

from probes.base import BaseProbe, ProbeReading


class PostprocessingProbe(BaseProbe):
    """Measures low-confidence flag rate and risk distribution."""

    def __init__(self, telemetry_store):
        super().__init__("postprocessing", telemetry_store)

    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        n = len(records)
        low_conf_count = stage_metadata.get("low_confidence_count", 0)
        low_conf_rate = round(low_conf_count / n, 4) if n > 0 else 0.0
        risk_dist = stage_metadata.get("risk_distribution", {})

        metrics = {
            "record_count": n,
            "low_confidence_flag_rate": low_conf_rate,
            "low_confidence_count": low_conf_count,
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
