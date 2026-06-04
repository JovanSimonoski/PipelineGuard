"""Synthetic patient data generator for pipeline testing."""

import random
import uuid
from datetime import datetime, timezone


CHIEF_COMPLAINTS = [
    "chest pain", "shortness of breath", "dizziness", "headache",
    "abdominal pain", "fever", "fatigue", "back pain", "nausea", "palpitations",
]

NORMAL_RANGES = {
    "age":                      (18, 90),
    "heart_rate":               (60.0, 100.0),
    "blood_pressure_systolic":  (90.0, 140.0),
    "blood_pressure_diastolic": (60.0, 90.0),
    "glucose_level":            (70.0, 140.0),
    "spo2":                     (94.0, 100.0),
    "temperature":              (36.1, 37.2),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normal_record() -> dict:
    return {
        "patient_id": f"P-{uuid.uuid4().hex[:8]}",
        "age": random.randint(18, 90),
        "heart_rate": round(random.uniform(60.0, 100.0), 1),
        "blood_pressure_systolic": round(random.uniform(90.0, 140.0), 1),
        "blood_pressure_diastolic": round(random.uniform(60.0, 90.0), 1),
        "glucose_level": round(random.uniform(70.0, 140.0), 1),
        "spo2": round(random.uniform(94.0, 100.0), 2),
        "temperature": round(random.uniform(36.1, 37.2), 2),
        "chief_complaint": random.choice(CHIEF_COMPLAINTS),
        "timestamp": _now_iso(),
    }


def _poisoned_record(record: dict) -> dict:
    """Inject subtle statistical drift into one or two vitals."""
    drift_fields = random.sample(
        ["heart_rate", "blood_pressure_systolic", "glucose_level", "temperature"], k=2
    )
    for field in drift_fields:
        lo, hi = NORMAL_RANGES[field]
        mean = (lo + hi) / 2
        std = (hi - lo) / 6
        # shift mean by 2-3 standard deviations
        shift = random.uniform(2.0, 3.0) * std * random.choice([-1, 1])
        record[field] = round(record[field] + shift, 2)
    return record


def _adversarial_record() -> dict:
    """Craft extreme values to push the model toward critical output."""
    return {
        "patient_id": f"P-{uuid.uuid4().hex[:8]}",
        "age": 92,
        "heart_rate": round(random.uniform(155.0, 180.0), 1),
        "blood_pressure_systolic": round(random.uniform(195.0, 220.0), 1),
        "blood_pressure_diastolic": round(random.uniform(115.0, 130.0), 1),
        "glucose_level": round(random.uniform(420.0, 600.0), 1),
        "spo2": round(random.uniform(72.0, 82.0), 2),
        "temperature": round(random.uniform(39.8, 41.2), 2),
        "chief_complaint": "chest pain",
        "timestamp": _now_iso(),
    }


def generate_batch(size: int = 20, mode: str = "normal") -> list[dict]:
    """
    Generate a batch of synthetic patient records.

    mode: "normal" | "poisoned" | "adversarial"
    """
    match mode:
        case "normal":
            return [_normal_record() for _ in range(size)]
        case "poisoned":
            records = [_normal_record() for _ in range(size)]
            # inject drift in ~30% of records
            poison_count = max(1, int(size * 0.30))
            indices = random.sample(range(size), poison_count)
            for i in indices:
                records[i] = _poisoned_record(records[i])
            return records
        case "adversarial":
            records = [_normal_record() for _ in range(size - 1)]
            records.append(_adversarial_record())
            return records
        case _:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'normal', 'poisoned', or 'adversarial'.")
