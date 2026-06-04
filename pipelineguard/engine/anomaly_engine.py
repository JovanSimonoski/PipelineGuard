"""Central anomaly detection coordinator."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from engine.baseline import BaselineManager
from engine.detectors.abuse import AbuseDetector
from engine.detectors.adversarial import AdversarialDetector
from engine.detectors.poisoning import PoisoningDetector
from probes.base import ProbeReading
from response.actions import ResponseOrchestrator


class ThreatType(str, Enum):
    POISONING = "data_poisoning"
    ADVERSARIAL = "adversarial_input"
    ABUSE = "access_abuse"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Represents a single detected anomaly with its context and response."""

    id: str
    threat_type: ThreatType
    severity: Severity
    stage: str
    batch_id: str
    timestamp: datetime
    score: float
    details: list[dict]
    response_triggered: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "stage": self.stage,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "details": self.details,
            "response_triggered": self.response_triggered,
        }


class AnomalyEngine:
    """Runs all detectors against every probe reading and coordinates responses."""

    _SEVERITY_THRESHOLDS = [
        (0.8, Severity.CRITICAL),
        (0.6, Severity.HIGH),
        (0.3, Severity.MEDIUM),
        (0.0, Severity.LOW),
    ]

    def __init__(self, sensitivity: str = "medium"):
        self.baseline = BaselineManager()
        self.poisoning = PoisoningDetector(sensitivity)
        self.adversarial = AdversarialDetector(sensitivity)
        self.abuse = AbuseDetector(sensitivity)
        self.alerts: list[AnomalyAlert] = []
        self.response = ResponseOrchestrator()

    def process(self, reading: ProbeReading) -> list[AnomalyAlert]:
        """
        Run all three detectors, generate alerts, trigger responses,
        and update baseline only for clean readings.
        """
        p_result = self.poisoning.detect(reading, self.baseline)
        a_result = self.adversarial.detect(reading, self.baseline)
        ab_result = self.abuse.detect(reading, self.baseline)

        new_alerts: list[AnomalyAlert] = []

        for threat_type, result in [
            (ThreatType.POISONING, p_result),
            (ThreatType.ADVERSARIAL, a_result),
            (ThreatType.ABUSE, ab_result),
        ]:
            if not result["detected"]:
                continue

            severity = self._score_to_severity(result["score"])
            response_action = self.response.handle(threat_type, severity, reading)

            alert = AnomalyAlert(
                id=uuid.uuid4().hex[:8],
                threat_type=threat_type,
                severity=severity,
                stage=reading.stage,
                batch_id=reading.batch_id,
                timestamp=datetime.now(timezone.utc),
                score=result["score"],
                details=result["details"],
                response_triggered=response_action,
            )
            self.alerts.append(alert)
            new_alerts.append(alert)

        # only feed clean readings into the baseline
        if not new_alerts:
            self.baseline.update(reading.stage, reading.metrics)

        return new_alerts

    def _score_to_severity(self, score: float) -> Severity:
        for threshold, severity in self._SEVERITY_THRESHOLDS:
            if score >= threshold:
                return severity
        return Severity.LOW

    def get_alerts(self, limit: int = 100) -> list[dict]:
        """Return most recent alerts as dicts, newest first."""
        return [a.to_dict() for a in reversed(self.alerts[-limit:])]

    def clear_alerts(self) -> None:
        self.alerts.clear()
        self.response.clear_quarantine()

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    @property
    def quarantine_count(self) -> int:
        return len(self.response._quarantined)
