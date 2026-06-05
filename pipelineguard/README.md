# PipelineGuard

A proof-of-concept **security-aware anomaly detection system** for a mock healthcare AI inference pipeline. PipelineGuard instruments every stage of a synthetic patient-triage workflow and watches for three classes of attack — **data poisoning**, **adversarial inputs**, and **access abuse** — using rolling statistical baselines and per-stage probes.

It ships with a FastAPI dashboard, a `tinyllama` (Ollama) inference backend with a deterministic fallback, and three injectable attack modes for live demonstrations.

> ⚠️ **Demo only.** Synthetic data, in-memory state, simulated responses, no clinical validity, no authentication. Do not expose to untrusted networks.

---

## Table of contents

1. [What it is, what it isn't](#what-it-is-what-it-isnt)
2. [Threat model](#threat-model)
3. [Architecture](#architecture)
4. [Project structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Quick start](#quick-start)
7. [Triggering attacks manually](#triggering-attacks-manually)
8. [API reference](#api-reference)
9. [Detector reference](#detector-reference)
10. [Response orchestrator](#response-orchestrator)
11. [Configuration](#configuration)
12. [Sensitivity tuning](#sensitivity-tuning)
13. [Development without Ollama](#development-without-ollama)
14. [Limitations](#limitations)

---

## What it is, what it isn't

| ✅ What it is | ❌ What it isn't |
|--------------|------------------|
| A pedagogical demonstration of telemetry-driven anomaly detection on an LLM pipeline | A production-grade security tool |
| A working FastAPI app with a live HTML dashboard and curl-driven attack injection | A real EHR integration |
| A statistical baseline + Z-score detector framework you can extend | An ML-based detector trained on labeled threats |
| Configurable across three sensitivity tiers | A tuned, false-positive-minimized system |

---

## Threat model

PipelineGuard monitors three threat classes mapped to OWASP-style attacks on ML pipelines:

| Threat | What it looks like in the data | Where it's detected |
|--------|--------------------------------|---------------------|
| **Data poisoning** | Subtle statistical drift in vitals — values shifted 2–3 σ from the clean baseline across ~30% of a batch | Any stage with numeric metrics that drift vs. the rolling baseline |
| **Adversarial input** | Crafted record(s) with extreme vitals designed to push the model into a wrong/low-confidence prediction | `inference` stage — confidence collapse, unknown-rate spike, variance explosion |
| **Access abuse** | Anomalous request patterns — batch-size spikes, abnormal preprocessing drop rate, normalization-violation flood | `collection` and `preprocessing` stages |

Each detector emits a 0–1 score; the engine maps scores to severities (`low`/`medium`/`high`/`critical`) and routes to the response orchestrator.

---

## Architecture

```
┌─────────────┬────────────────┬──────────────┬───────────────────┬────────────────┐
│ Collection  │ Preprocessing  │  Inference   │  Post-processing  │ Output Delivery│
│             │                │   (Ollama)   │                   │                │
└──────┬──────┴───────┬────────┴──────┬───────┴─────────┬─────────┴────────┬───────┘
       │              │               │                 │                  │
       ▼              ▼               ▼                 ▼                  ▼
   ┌──────┐      ┌──────┐         ┌──────┐          ┌──────┐           ┌──────┐
   │Probe │      │Probe │         │Probe │          │Probe │           │Probe │
   └──┬───┘      └──┬───┘         └──┬───┘          └──┬───┘           └──┬───┘
      └─────────────┴────────────────┴──────────────────┴──────────────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │ TelemetryStore   │  (in-memory, deque/stage, capped 500)
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐         ┌──────────────────┐
                           │  AnomalyEngine   │◀───────▶│ BaselineManager  │
                           └────────┬─────────┘         │ (30-reading window│
                                    │                   │  per stage/metric)│
            ┌───────────────────────┼──────────────────┐└──────────────────┘
            ▼                       ▼                  ▼
     ┌─────────────┐       ┌──────────────┐      ┌─────────┐
     │ Poisoning   │       │ Adversarial  │      │ Abuse   │
     │ Detector    │       │ Detector     │      │ Detector│
     └──────┬──────┘       └──────┬───────┘      └────┬────┘
            └───────────────┬─────┴───────────────────┘
                            ▼
                  ┌────────────────────┐
                  │ResponseOrchestrator│
                  │ quarantine /       │
                  │ rollback (sim) /   │
                  │ alert              │
                  └────────────────────┘
```

### Per-run sequence

1. `PipelineRunner.run(mode)` generates a batch of 20 synthetic records via `data_generator.generate_batch(mode)`.
2. For each of the five stages:
   1. Stage function transforms records and emits `meta`.
   2. The stage's probe builds a `ProbeReading` from `records + meta` and appends it to `TelemetryStore`.
   3. `AnomalyEngine.process(reading)` runs all three detectors.
      - If any detector fires, the engine maps score → severity, dispatches `ResponseOrchestrator.handle(...)`, and creates an `AnomalyAlert`.
      - If **no** detector fires, the reading is added to the rolling baseline (so only clean traffic teaches the baseline).
   4. If the batch was quarantined at this stage, the loop breaks and remaining stages are skipped.

### Baseline & warm-up

On startup the runner executes **5 warm-up runs in `normal` mode** before any traffic is served. This seeds the baseline so detectors have something to compare against. Without warm-up, the first few real batches would fire false positives (no baseline → division-by-near-zero).

The baseline window holds 30 readings per `(stage, metric)`. A metric needs ≥ 10 samples before it's considered established, and ≥ 5 before stats are returned.

---

## Project structure

```
pipelineguard/
├── api/
│   └── app.py                   # FastAPI app, dashboard HTML, REST endpoints, lifespan
├── pipeline/
│   ├── data_generator.py        # normal / poisoned / adversarial batch generators
│   ├── stages.py                # 5 stage functions (collection → output_delivery)
│   └── runner.py                # PipelineRunner — orchestrates a single end-to-end run
├── probes/
│   ├── base.py                  # ProbeReading dataclass + BaseProbe ABC
│   ├── collection.py            # one probe per stage; each extracts numeric telemetry
│   ├── preprocessing.py
│   ├── inference.py
│   ├── postprocessing.py
│   └── output.py
├── store/
│   └── telemetry.py             # in-memory deque (500/stage) of ProbeReadings
├── engine/
│   ├── baseline.py              # rolling-window stats per (stage, metric)
│   ├── anomaly_engine.py        # runs all detectors, scores → severity, routes responses
│   └── detectors/
│       ├── poisoning.py         # Z-score on any numeric metric
│       ├── adversarial.py       # confidence-collapse / unknown-rate / variance
│       └── abuse.py             # batch-size / drop-rate / normalization-violation spikes
├── response/
│   └── actions.py               # ResponseOrchestrator — quarantine / rollback / alert
├── ollama_client/
│   └── client.py                # thin httpx wrapper around /api/chat
├── Dockerfile
├── docker-compose.yml           # pipelineguard + ollama services
└── requirements.txt
```

---

## Prerequisites

- Docker and Docker Compose v2+
- ~4 GB free disk space (for the Ollama `tinyllama` model)
- Ports `8000` (FastAPI) and `11434` (Ollama) free

---

## Quick start

```bash
git clone https://github.com/JovanSimonoski/PipelineGuard.git
cd PipelineGuard/pipelineguard
docker compose up --build
```

Wait approximately 2 minutes for Ollama to pull `tinyllama` and the 5-run warm-up to complete, then open:

```
http://localhost:8000
```

The dashboard auto-refreshes every 5 seconds. Use the buttons on the page or the `curl` commands below to inject attacks.

---

## Triggering attacks manually

```bash
# Normal run (should produce 0 alerts after warm-up)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"mode":"normal"}'

# Data poisoning (subtle statistical drift injected into ~30% of records)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"mode":"poisoned"}'

# Adversarial input (one extreme record crafted to confuse the model)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"mode":"adversarial"}'

# View current alerts
curl http://localhost:8000/alerts

# Clear alerts and quarantine state (demo reset)
curl -X DELETE http://localhost:8000/alerts

# Health check
curl http://localhost:8000/health
```

### What each mode looks like in the data

| Mode | Generator behavior | Expected detectors |
|------|--------------------|--------------------|
| `normal` | 20 records sampled from clinically-normal vital ranges | none |
| `poisoned` | 30% of records have 2 vitals shifted by 2–3 σ | `PoisoningDetector` (drift on preprocessing/postprocessing metrics) |
| `adversarial` | 1 extreme record (HR 155+, glucose 420+, SpO₂ <82) added to 19 normal | `AdversarialDetector` (confidence collapse, variance spike) |

---

## API reference

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| `GET` | `/` | — | Auto-refreshing HTML dashboard |
| `POST` | `/run` | `{"mode": "normal" \| "poisoned" \| "adversarial"}` | Trigger one end-to-end pipeline run; returns a run summary |
| `POST` | `/run/auto` | `{"enabled": true \| false}` | Enable/disable the background APScheduler that fires `normal` runs every `PIPELINE_RUN_INTERVAL` seconds |
| `GET` | `/alerts` | — | Up to 100 most recent `AnomalyAlert`s (newest first) |
| `GET` | `/alerts/{id}` | — | One alert by 8-char id |
| `GET` | `/telemetry/{stage}` | — | Last 20 `ProbeReading`s for `stage` ∈ `collection`, `preprocessing`, `inference`, `postprocessing`, `output_delivery` |
| `GET` | `/health` | — | `{status, ollama, model, sensitivity, auto_run}` |
| `DELETE` | `/alerts` | — | Wipe alerts and quarantine set (demo reset) |

---

## Detector reference

All three detectors run on every probe reading. Each returns `{detected, score (0-1), details[]}`. The engine combines scores per detector → severity bucket → response action.

### `PoisoningDetector`
Compares every numeric metric in the reading against the rolling baseline using Z-score.

- Trigger: `|z| > threshold` (threshold depends on sensitivity tier)
- Score: mean of `min(|z|/threshold, 1.0)` across all metrics with sufficient baseline
- Why it catches `poisoned`: drifted vitals shift the means and stds of preprocessing/postprocessing metrics enough to exceed the rolling baseline's variance.

### `AdversarialDetector`
Only fires at the `inference` stage. Four checks:

1. **Absolute low confidence** — `avg_confidence < threshold`
2. **Confidence drop vs. baseline** — Z-score of avg confidence > 2.0
3. **High unknown rate** — `unknown_risk_rate > threshold`
4. **High confidence variance** — Z-score of confidence std > multiplier

Why it catches `adversarial`: an extreme record either pushes the model to a low-confidence "unknown" output or spikes the in-batch confidence variance.

### `AbuseDetector`
Three checks:

1. **Batch-size spike** — `record_count` ratio vs. baseline mean exceeds `spike_factor` (or its inverse)
2. **Drop-rate spike** (preprocessing only) — Z-score of drop rate > 2.5
3. **Normalization-violation spike** (preprocessing only) — Z-score of violation rate > 2.5

Not exercised by the built-in attack modes — you'd trigger it by sending an oversized batch or a batch with many malformed records.

---

## Response orchestrator

Severity → action mapping (in [response/actions.py](response/actions.py)):

| Severity | Score range | Actions taken |
|----------|-------------|---------------|
| `critical` | ≥ 0.8 | quarantine batch + simulated rollback + structured alert |
| `high` | ≥ 0.6 | quarantine batch + structured alert |
| `medium` | ≥ 0.3 | structured alert |
| `low` | ≥ 0.0 | log only |

**Quarantine** is the only response with teeth: the runner checks `is_quarantined(batch_id)` after each stage and stops the pipeline if true. Rollback and alerts are console/in-memory only.

---

## Configuration

All configuration is via environment variables in `docker-compose.yml`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `tinyllama` | Model id pulled at startup |
| `PIPELINE_RUN_INTERVAL` | `8` | Seconds between background `normal` runs |
| `ANOMALY_SENSITIVITY` | `medium` | `low` / `medium` / `high` — see table below |
| `SKIP_INFERENCE` | `false` | If `true`, skip Ollama and use the deterministic mock |

---

## Sensitivity tuning

| Tier | Z-score threshold (poisoning) | Confidence floor (adversarial) | Spike factor (abuse) |
|------|------------------------------|--------------------------------|----------------------|
| `low` | 3.5 | 0.20 | 5× |
| `medium` (default) | 2.5 | 0.35 | 3× |
| `high` | 1.5 | 0.50 | 2× |

Higher sensitivity = more alerts, higher false-positive rate. Use `low` for demos with lots of `normal` traffic, `high` to make subtle attacks pop.

---

## Development without Ollama

Set `SKIP_INFERENCE=true` in `docker-compose.yml` to bypass Ollama entirely. The mock in [`pipeline/stages.py`](pipeline/stages.py) `_mock_inference()` assigns `risk_level` deterministically from raw vitals:

| Condition | risk_level | confidence |
|-----------|-----------|------------|
| `glucose > 400` or `spo2 < 80` or `hr > 150` | `critical` | 0.75–0.95 |
| `glucose > 200` or `spo2 < 90` or `hr > 120` | `high` | 0.65–0.85 |
| `glucose > 140` or `hr > 100` | `medium` | 0.55–0.80 |
| otherwise | `low` | 0.70–0.95 |

This makes pipeline runs near-instant for detector development.

---

## Limitations

- **In-memory only** — baselines, alerts, telemetry, and quarantine are all lost on restart.
- **No real clinical data** — synthetic vitals only; risk classifications have no medical validity.
- **Single-process** — APScheduler runs in the same process as the API; no horizontal scaling.
- **Tiny model** — `tinyllama` is a toy. Inference quality is irrelevant to the security demonstration.
- **Simulated responses** — quarantine is enforced only inside the runner loop; rollback and alerts are console/in-memory.
- **No authentication** — the API is completely open; do not expose it to untrusted networks.
- **Detector logic is rule-based** — no learned threat classifier. Sensitivity tuning is the only knob.
