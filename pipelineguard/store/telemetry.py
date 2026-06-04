"""In-memory time-series telemetry store."""

from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probes.base import ProbeReading


class TelemetryStore:
    """Thread-safe in-memory store of ProbeReading objects, capped per stage."""

    def __init__(self, maxlen: int = 500):
        self._maxlen = maxlen
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._maxlen))

    def append(self, reading: "ProbeReading") -> None:
        """Add a reading to its stage bucket."""
        self._store[reading.stage].append(reading)

    def get_recent(self, stage: str, n: int = 50) -> list:
        """Return up to n most recent readings for a stage (newest last)."""
        bucket = self._store.get(stage, deque())
        items = list(bucket)
        return items[-n:] if len(items) > n else items

    def get_all_stages(self) -> list[str]:
        """Return all stage names that have at least one reading."""
        return [s for s, d in self._store.items() if d]

    def get_latest(self, stage: str):
        """Return the most recent reading for a stage, or None."""
        bucket = self._store.get(stage, deque())
        return bucket[-1] if bucket else None

    def total_readings(self) -> int:
        return sum(len(d) for d in self._store.values())
