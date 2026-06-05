"""Five mock pipeline stages for the healthcare AI pipeline."""

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

from rich.console import Console

console = Console()

REQUIRED_FIELDS = [
    "patient_id", "age", "heart_rate", "blood_pressure_systolic",
    "blood_pressure_diastolic", "glucose_level", "spo2", "temperature",
    "chief_complaint", "timestamp",
]

NORMALIZATION_BOUNDS: dict[str, tuple[float, float]] = {
    "age":                      (0.0,   120.0),
    "heart_rate":               (30.0,  220.0),
    "blood_pressure_systolic":  (50.0,  250.0),
    "blood_pressure_diastolic": (30.0,  150.0),
    "glucose_level":            (20.0,  700.0),
    "spo2":                     (50.0,  100.0),
    "temperature":              (34.0,  43.0),
}

TRIAGE_SYSTEM_PROMPT = (
    "You are a clinical triage assistant. Given patient vitals, respond with ONLY a JSON object:\n"
    '{"risk_level": "low|medium|high|critical", "confidence": 0.0-1.0, "flags": ["list of concerns"]}\n'
    "Do not include any explanation or markdown."
)

SKIP_INFERENCE = os.getenv("SKIP_INFERENCE", "false").lower() == "true"


def _mock_inference(record: dict) -> dict:
    """Fast mock inference used when SKIP_INFERENCE=true or Ollama is unavailable."""
    glucose = record.get("glucose_level", 100)
    spo2 = record.get("spo2", 98)
    hr = record.get("heart_rate", 75)

    if glucose > 400 or spo2 < 80 or hr > 150:
        risk_level = "critical"
        confidence = round(random.uniform(0.75, 0.95), 2)
    elif glucose > 200 or spo2 < 90 or hr > 120:
        risk_level = "high"
        confidence = round(random.uniform(0.65, 0.85), 2)
    elif glucose > 140 or hr > 100:
        risk_level = "medium"
        confidence = round(random.uniform(0.55, 0.80), 2)
    else:
        risk_level = "low"
        confidence = round(random.uniform(0.70, 0.95), 2)

    return {"risk_level": risk_level, "confidence": confidence, "flags": []}


def stage_collection(raw_records: list[dict]) -> tuple[list[dict], dict]:
    """Simulate receiving records from a sensor/EHR system."""
    batch_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for rec in raw_records:
        r = dict(rec)
        r["received_at"] = now
        r["batch_id"] = batch_id
        records.append(r)

    metadata = {
        "source": "mock_ehr",
        "record_count": len(records),
        "batch_id": batch_id,
    }
    console.print(f"[Stage] collection → {len(records)} records | OK", style="dim")
    return records, metadata


def stage_preprocessing(records: list[dict]) -> tuple[list[dict], dict]:
    """Normalize vitals to 0-1 and drop incomplete records."""
    clean = []
    dropped = 0
    normalization_violations = 0

    for rec in records:
        if not all(f in rec for f in REQUIRED_FIELDS):
            dropped += 1
            continue
        r = dict(rec)
        for field, (lo, hi) in NORMALIZATION_BOUNDS.items():
            if field in r and isinstance(r[field], (int, float)):
                val = r[field]
                if val < lo or val > hi:
                    normalization_violations += 1
                r[f"{field}_norm"] = round((val - lo) / (hi - lo), 4)
        r["preprocessed"] = True
        clean.append(r)

    metadata = {
        "dropped_count": dropped,
        "normalized_fields": list(NORMALIZATION_BOUNDS.keys()),
        "normalization_violations": normalization_violations,
        "record_count": len(clean),
    }
    console.print(
        f"[Stage] preprocessing → {len(clean)} records ({dropped} dropped) | OK",
        style="dim",
    )
    return clean, metadata


def stage_inference(records: list[dict], ollama_client) -> tuple[list[dict], dict]:
    """Run triage inference on each record via Ollama (or mock)."""
    t_start = time.monotonic()
    results = []
    confidences = []

    for rec in records:
        vitals = {
            k: rec.get(k)
            for k in [
                "age", "heart_rate", "blood_pressure_systolic",
                "blood_pressure_diastolic", "glucose_level", "spo2", "temperature",
                "chief_complaint",
            ]
        }
        user_msg = json.dumps(vitals)

        inference_result = None

        if not SKIP_INFERENCE and ollama_client is not None:
            try:
                raw = ollama_client.generate(TRIAGE_SYSTEM_PROMPT, user_msg, timeout=60.0)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                raw = raw.split("\n")[0].strip()
                parsed = json.loads(raw)
                inference_result = parsed[0] if isinstance(parsed, list) else parsed
                if not isinstance(inference_result, dict):
                    raise ValueError(f"Unexpected inference result type: {type(inference_result)}")
            except Exception as exc:
                console.print(f"[yellow]Ollama inference failed: {exc}. Using mock.[/yellow]")

        if inference_result is None:
            inference_result = _mock_inference(rec)

        r = dict(rec)
        r["inference_result"] = inference_result
        results.append(r)
        confidences.append(inference_result.get("confidence", 0.0))

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    metadata = {
        "model_used": os.getenv("OLLAMA_MODEL", "tinyllama"),
        "avg_confidence": avg_conf,
        "inference_time_ms": elapsed_ms,
        "record_count": len(results),
        "skip_inference": SKIP_INFERENCE,
    }
    console.print(
        f"[Probe] inference | avg_confidence={avg_conf} | records={len(results)}",
        style="dim",
    )
    return results, metadata


def stage_postprocessing(records: list[dict]) -> tuple[list[dict], dict]:
    """Aggregate inference results and flag low-confidence outputs."""
    risk_dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0, "unknown": 0}
    low_conf_count = 0

    processed = []
    for rec in records:
        r = dict(rec)
        ir = r.get("inference_result", {})
        risk = ir.get("risk_level", "unknown")
        conf = ir.get("confidence", 0.0)

        risk_dist[risk] = risk_dist.get(risk, 0) + 1

        if conf < 0.4:
            r["low_confidence_flag"] = True
            low_conf_count += 1

        r["post_processed"] = True
        processed.append(r)

    metadata = {
        "risk_distribution": risk_dist,
        "low_confidence_count": low_conf_count,
        "record_count": len(processed),
    }
    console.print(
        f"[Stage] post-processing → risk_dist={risk_dist} | OK",
        style="dim",
    )
    return processed, metadata


def stage_output_delivery(records: list[dict]) -> tuple[list[dict], dict]:
    """Simulate writing results to a clinical UI."""
    delivered = 0
    for rec in records:
        ir = rec.get("inference_result", {})
        console.print(
            f"  [Output] patient={rec.get('patient_id')} "
            f"risk={ir.get('risk_level','?')} conf={ir.get('confidence','?')}",
            style="dim cyan",
        )
        delivered += 1

    metadata = {
        "delivered_count": delivered,
        "delivery_target": "mock_clinical_ui",
    }
    console.print(f"[Stage] output_delivery → {delivered} records delivered | OK", style="dim")
    return records, metadata
