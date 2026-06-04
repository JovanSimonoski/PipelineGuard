"""Probe for the Preprocessing pipeline stage."""

from datetime import datetime, timezone

from probes.base import BaseProbe, ProbeReading


class PreprocessingProbe(BaseProbe):
    """Measures drop rate and normalization violations."""

    def __init__(self, telemetry_store):
        super().__init__("preprocessing", telemetry_store)

    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        n = len(records)
        dropped = stage_metadata.get("dropped_count", 0)
        total = n + dropped

        drop_rate = round(dropped / total, 4) if total > 0 else 0.0
        violations = stage_metadata.get("normalization_violations", 0)
        violation_rate = round(violations / n, 4) if n > 0 else 0.0

        metrics = {
            "record_count": n,
            "drop_rate": drop_rate,
            "dropped_count": dropped,
            "normalization_violation_rate": violation_rate,
            "normalization_violations": violations,
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
