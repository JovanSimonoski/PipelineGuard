"""Orchestrates a full end-to-end pipeline run."""

import time
import uuid
from datetime import datetime, timezone

from rich.console import Console

from pipeline.data_generator import generate_batch
from pipeline.stages import (
    stage_collection,
    stage_inference,
    stage_output_delivery,
    stage_postprocessing,
    stage_preprocessing,
)
from probes.collection import CollectionProbe
from probes.inference import InferenceProbe
from probes.output import OutputDeliveryProbe
from probes.postprocessing import PostprocessingProbe
from probes.preprocessing import PreprocessingProbe

console = Console()


class PipelineRunner:
    """Runs the full 5-stage pipeline and feeds readings to the anomaly engine."""

    def __init__(self, engine, telemetry, ollama_client):
        self.engine = engine
        self.telemetry = telemetry
        self.ollama = ollama_client
        self.run_count = 0

        self._probes = {
            "collection":      CollectionProbe(telemetry),
            "preprocessing":   PreprocessingProbe(telemetry),
            "inference":       InferenceProbe(telemetry),
            "postprocessing":  PostprocessingProbe(telemetry),
            "output_delivery": OutputDeliveryProbe(telemetry),
        }

    def run(self, attack_mode: str = "normal") -> dict:
        """
        Execute all five pipeline stages, probe each stage, and run anomaly detection.

        Returns a run-summary dict.
        """
        self.run_count += 1
        run_id = uuid.uuid4().hex[:8]
        t_start = time.monotonic()

        console.print(
            f"\n[Pipeline] Run #{self.run_count} | mode={attack_mode} | run_id={run_id}",
            style="bold blue",
        )

        raw_records = generate_batch(size=20, mode=attack_mode)

        all_alerts: list = []
        stages_completed = 0
        quarantined = False
        batch_id: str | None = None

        stage_fns = [
            ("collection",      lambda r: stage_collection(r)),
            ("preprocessing",   lambda r: stage_preprocessing(r)),
            ("inference",       lambda r: stage_inference(r, self.ollama)),
            ("postprocessing",  lambda r: stage_postprocessing(r)),
            ("output_delivery", lambda r: stage_output_delivery(r)),
        ]

        records = raw_records
        for stage_name, stage_fn in stage_fns:
            records, meta = stage_fn(records)

            if batch_id is None:
                batch_id = meta.get("batch_id", run_id)

            probe = self._probes[stage_name]
            reading = probe.collect(records, meta)
            probe.emit(reading)

            new_alerts = self.engine.process(reading)
            all_alerts.extend(new_alerts)
            stages_completed += 1

            for alert in new_alerts:
                console.print(
                    f"[ALERT] ⚠  {alert.threat_type.value.upper()} | "
                    f"stage={alert.stage} | severity={alert.severity.value.upper()} | "
                    f"score={alert.score}",
                    style="bold red",
                )

            if self.engine.response.is_quarantined(batch_id):
                console.print(
                    f"[Pipeline] Batch {batch_id} quarantined — stopping at stage {stage_name}",
                    style="bold yellow",
                )
                quarantined = True
                break

        duration_ms = int((time.monotonic() - t_start) * 1000)

        return {
            "run_id": run_id,
            "attack_mode": attack_mode,
            "stages_completed": stages_completed,
            "alerts_triggered": [
                {
                    "id": a.id,
                    "threat_type": a.threat_type.value,
                    "severity": a.severity.value,
                    "stage": a.stage,
                    "score": a.score,
                }
                for a in all_alerts
            ],
            "quarantined": quarantined,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
