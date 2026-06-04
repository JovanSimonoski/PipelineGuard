"""Base probe abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ProbeReading:
    """Snapshot of telemetry collected from a single pipeline stage execution."""

    stage: str
    timestamp: datetime
    batch_id: str
    record_count: int
    metrics: dict          # numeric telemetry (means, stds, rates)
    metadata: dict         # pass-through from stage_metadata
    raw_sample: list[dict] = field(default_factory=list)  # first 3 records for inspection


class BaseProbe(ABC):
    """Abstract base for all stage probes."""

    def __init__(self, stage_name: str, telemetry_store):
        self.stage_name = stage_name
        self.store = telemetry_store

    @abstractmethod
    def collect(self, records: list[dict], stage_metadata: dict) -> ProbeReading:
        """Extract a ProbeReading from the stage output."""
        ...

    def emit(self, reading: ProbeReading) -> None:
        """Persist the reading to the telemetry store."""
        self.store.append(reading)
