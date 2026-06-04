# PipelineGuard

A proof-of-concept security-aware anomaly detection system for a mock healthcare AI data pipeline.

## Prerequisites

- Docker and Docker Compose v2+
- ~4 GB free disk space (for the Ollama `tinyllama` model)
- No other process on port `8000` or `11434`

## Quick Start

```bash
git clone <repo> && cd pipelineguard
docker compose up --build
```

Wait approximately 2 minutes for Ollama to download `tinyllama` and for the 15-run baseline warm-up to complete, then open:

```
http://localhost:8000
```

The dashboard auto-refreshes every 5 seconds. Use the buttons on the page or the curl commands below to inject attacks.

## Triggering Attacks Manually

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

# Clear alerts and quarantine state
curl -X DELETE http://localhost:8000/alerts

# Health check
curl http://localhost:8000/health
```

## Architecture

```
Collection → Preprocessing → Inference → Post-processing → Output Delivery
     ↓              ↓             ↓              ↓                ↓
  Probe           Probe         Probe          Probe            Probe
     └──────────────┴─────────────┴──────────────┴────────────────┘
                                  ↓
                          Anomaly Engine
                     ┌─────────┬─────────┐
                 Poisoning  Adversarial  Abuse
                     └─────────┴─────────┘
                                  ↓
                        Response Orchestrator
                   (Quarantine / Rollback / Alert)
```

**Pipeline stages** (`pipeline/stages.py`) — five sequential transforms on synthetic patient records. Each returns the transformed batch plus metadata consumed by its probe.

**Data generator** (`pipeline/data_generator.py`) — produces batches of 20 synthetic patient vitals in three modes: `normal`, `poisoned` (mean-shifted vitals), or `adversarial` (one extreme record).

**Probes** (`probes/`) — one probe per stage. Each `collect()` call extracts numeric telemetry metrics (means, rates, distributions) from the stage output and emits a `ProbeReading` to the telemetry store.

**Telemetry store** (`store/telemetry.py`) — in-memory `deque` per stage (capped at 500 readings). No persistence — restarts clean.

**Baseline manager** (`engine/baseline.py`) — sliding window of the last 30 clean readings per (stage, metric) pair. Only updated when a reading passes all detectors.

**Detectors** (`engine/detectors/`):
- `PoisoningDetector` — Z-score drift on any numeric metric vs. rolling baseline.
- `AdversarialDetector` — Confidence collapse, unknown-rate spike, and variance explosion at the inference stage.
- `AbuseDetector` — Batch-size spike and preprocessing drop-rate spike.

**Anomaly engine** (`engine/anomaly_engine.py`) — runs all three detectors per reading, maps score → severity (low / medium / high / critical), routes to `ResponseOrchestrator`, accumulates `AnomalyAlert` objects.

**Response orchestrator** (`response/actions.py`) — quarantine (blocks further pipeline stages for the batch), simulated rollback (log only), and structured alert logging.

**Ollama client** (`ollama_client/client.py`) — thin `httpx` wrapper around `/api/chat`. Falls back to a deterministic mock when `SKIP_INFERENCE=true` or Ollama is unavailable.

**FastAPI app** (`api/app.py`) — serves the auto-refreshing HTML dashboard and REST endpoints. Uses APScheduler for background auto-runs.

## Sensitivity Tuning

Set the `ANOMALY_SENSITIVITY` environment variable in `docker-compose.yml`:

| Value | Z-score threshold | Confidence threshold | Spike factor |
|-------|------------------|---------------------|-------------|
| `low` | 3.5 | 0.20 | 5× |
| `medium` (default) | 2.5 | 0.35 | 3× |
| `high` | 1.5 | 0.50 | 2× |

Higher sensitivity = more alerts, higher false-positive rate.

## Development Without Ollama

Set `SKIP_INFERENCE=true` in `docker-compose.yml` to bypass Ollama entirely. A deterministic mock assigns risk levels based on raw vital values, making the pipeline run fast for detector development and testing.

## Limitations / What This Is Not

- **In-memory only** — all state (baselines, alerts, telemetry) is lost on restart.
- **No real clinical data** — synthetic vitals only; results have no medical validity.
- **Single-process** — no horizontal scaling; APScheduler runs in the same process as the API.
- **Tiny model** — `tinyllama` is a toy model unsuitable for real triage; inference quality is irrelevant to the security demonstration.
- **Simulated responses** — quarantine, rollback, and alerts are all in-memory/console; no actual pipeline gates are enforced.
- **No authentication** — the API is completely open; do not expose it to untrusted networks.
