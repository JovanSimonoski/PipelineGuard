"""Automated response actions: quarantine, rollback, and alerting."""

from datetime import datetime, timezone

from rich.console import Console

console = Console()


class ResponseOrchestrator:
    """Routes anomaly alerts to appropriate response actions."""

    def __init__(self):
        self._quarantined: set[str] = set()
        self._alert_log: list[dict] = []

    def handle(self, threat_type, severity, reading) -> str:
        """
        Route to response actions based on severity.

        CRITICAL  → quarantine + rollback + alert
        HIGH      → quarantine + alert
        MEDIUM    → alert
        LOW       → log only

        Returns the primary action taken.
        """
        alert_data = {
            "threat_type": threat_type.value,
            "severity": severity.value,
            "stage": reading.stage,
            "batch_id": reading.batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        match severity.value:
            case "critical":
                self.quarantine_batch(reading.batch_id)
                self.trigger_rollback(reading.stage)
                self.send_alert(alert_data)
                primary_action = "quarantine+rollback+alert"
            case "high":
                self.quarantine_batch(reading.batch_id)
                self.send_alert(alert_data)
                primary_action = "quarantine+alert"
            case "medium":
                self.send_alert(alert_data)
                primary_action = "alert"
            case _:
                console.print(
                    f"[Response] LOG: {threat_type.value} at {reading.stage} "
                    f"(severity=low, score below action threshold)",
                    style="dim yellow",
                )
                primary_action = "log"

        return primary_action

    def quarantine_batch(self, batch_id: str) -> None:
        """Mark a batch as quarantined — it will not proceed further in the pipeline."""
        self._quarantined.add(batch_id)
        console.print(
            f"[Response] QUARANTINE triggered for batch {batch_id}",
            style="bold yellow",
        )

    def trigger_rollback(self, stage: str) -> None:
        """Simulate a rollback to the last clean baseline snapshot."""
        console.print(
            f"[Response] ROLLBACK: would revert {stage} to last clean baseline snapshot",
            style="bold magenta",
        )

    def send_alert(self, alert_data: dict) -> None:
        """Log a structured alert event and append to the in-memory alert log."""
        self._alert_log.append(alert_data)
        console.print(
            f"[Response] ALERT logged: threat={alert_data['threat_type']} | "
            f"stage={alert_data['stage']} | severity={alert_data['severity']}",
            style="bold red",
        )

    def is_quarantined(self, batch_id: str) -> bool:
        return batch_id in self._quarantined

    def clear_quarantine(self) -> None:
        self._quarantined.clear()

    def get_alert_log(self) -> list[dict]:
        return list(self._alert_log)
