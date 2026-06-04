"""Probe for the Collection pipeline stage."""

from datetime import datetime, timezone

import numpy as np

from probes.base import BaseProbe, ProbeReading

NUMERIC_FIELDS = [
    "age", "heart_rate", "blood_pressure_systolic",
    "blood_pressure_diastolic", "glucose_level", "spo2", "temperature",
]

ALL_FIELDS = NUMERIC_FIELDS + ["patient_id", "chief_complaint", "timestamp"]


class CollectionProbe(BaseProbe):
    """Measures field completeness and value range distribution at ingestion."""

    def __init__(self, telemetry_store):
        super().__init__("collection", telemetry_store)

    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        n = len(records)
        metrics: dict = {"record_count": n}

        if n > 0:
            completeness = sum(
                sum(1 for f in ALL_FIELDS if f in r) / len(ALL_FIELDS)
                for r in records
            ) / n
            metrics["field_completeness_rate"] = round(completeness, 4)

            for field in NUMERIC_FIELDS:
                vals = [r[field] for r in records if isinstance(r.get(field), (int, float))]
                if vals:
                    metrics[f"{field}_mean"] = round(float(np.mean(vals)), 3)
                    metrics[f"{field}_std"] = round(float(np.std(vals)), 3)
                    metrics[f"{field}_min"] = round(float(np.min(vals)), 3)
                    metrics[f"{field}_max"] = round(float(np.max(vals)), 3)

        return ProbeReading(
            stage=self.stage_name,
            timestamp=datetime.now(timezone.utc),
            batch_id=stage_metadata.get("batch_id", ""),
            record_count=n,
            metrics=metrics,
            metadata=stage_metadata,
            raw_sample=records[:3],
        )
