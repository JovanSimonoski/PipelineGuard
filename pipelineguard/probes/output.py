"""Probe for the Output Delivery pipeline stage."""

from datetime import datetime, timezone

from probes.base import BaseProbe, ProbeReading


class OutputDeliveryProbe(BaseProbe):
    """Measures delivery success rate and record count."""

    def __init__(self, telemetry_store):
        super().__init__("output_delivery", telemetry_store)

    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        n = len(records)
        delivered = stage_metadata.get("delivered_count", n)
        success_rate = round(delivered / n, 4) if n > 0 else 0.0

        metrics = {
            "record_count": n,
            "delivered_count": delivered,
            "delivery_success_rate": success_rate,
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
