"""FastAPI application: REST endpoints + live HTML dashboard."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from rich.console import Console

from engine.anomaly_engine import AnomalyEngine
from ollama_client.client import OllamaClient
from pipeline.runner import PipelineRunner
from store.telemetry import TelemetryStore

console = Console()

SENSITIVITY = os.getenv("ANOMALY_SENSITIVITY", "medium")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
RUN_INTERVAL = int(os.getenv("PIPELINE_RUN_INTERVAL", "8"))
WARMUP_RUNS = 15


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    telemetry = TelemetryStore()
    engine = AnomalyEngine(sensitivity=SENSITIVITY)
    ollama = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL)
    runner = PipelineRunner(engine=engine, telemetry=telemetry, ollama_client=ollama)

    app.state.telemetry = telemetry
    app.state.engine = engine
    app.state.ollama = ollama
    app.state.runner = runner
    app.state.scheduler = None
    app.state.auto_run_enabled = False

    # Warm up the baseline with normal runs before serving traffic
    console.print(
        f"\n[PipelineGuard] Warming up baseline with {WARMUP_RUNS} normal runs...",
        style="bold cyan",
    )
    for i in range(WARMUP_RUNS):
        runner.run(attack_mode="normal")
        if (i + 1) % 5 == 0:
            console.print(f"[PipelineGuard] Warmup progress: {i + 1}/{WARMUP_RUNS}", style="cyan")

    console.print("[PipelineGuard] Baseline warm-up complete. Starting auto-run scheduler.", style="bold green")

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: runner.run(attack_mode="normal"),
        "interval",
        seconds=RUN_INTERVAL,
        id="auto_run",
    )
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.auto_run_enabled = True

    yield

    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title="PipelineGuard", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    mode: str = "normal"


class AutoRunRequest(BaseModel):
    enabled: bool


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

def _build_dashboard(engine: AnomalyEngine, telemetry: TelemetryStore, runner: PipelineRunner, auto_enabled: bool) -> str:
    recent_alerts = engine.get_alerts(10)
    total_alerts = engine.alert_count
    quarantine_count = engine.quarantine_count
    total_runs = runner.run_count

    # pipeline status
    last_10 = engine.get_alerts(10)
    if not last_10:
        status_color = "#22c55e"
        status_text = "HEALTHY"
    elif any(a["severity"] in ("critical", "high") for a in last_10[:3]):
        status_color = "#ef4444"
        status_text = "CRITICAL"
    else:
        status_color = "#f59e0b"
        status_text = "WARNING"

    # per-stage telemetry
    stage_rows = ""
    for stage in ["collection", "preprocessing", "inference", "postprocessing", "output_delivery"]:
        latest = telemetry.get_latest(stage)
        if latest:
            key_metrics = {k: v for k, v in latest.metrics.items() if isinstance(v, (int, float))}
            top_metrics = list(key_metrics.items())[:4]
            metrics_str = " | ".join(f"{k}={v}" for k, v in top_metrics)
            stage_rows += f"<tr><td>{stage}</td><td>{latest.record_count}</td><td style='font-size:0.75rem'>{metrics_str}</td></tr>"
        else:
            stage_rows += f"<tr><td>{stage}</td><td>—</td><td style='color:#666'>no data yet</td></tr>"

    # alert rows
    severity_colors = {"low": "#6b7280", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}
    alert_rows = ""
    for a in recent_alerts:
        color = severity_colors.get(a["severity"], "#6b7280")
        alert_rows += (
            f"<tr>"
            f"<td>{a['id']}</td>"
            f"<td>{a['stage']}</td>"
            f"<td>{a['threat_type']}</td>"
            f"<td style='color:{color};font-weight:bold'>{a['severity'].upper()}</td>"
            f"<td>{a['score']}</td>"
            f"<td style='font-size:0.75rem'>{a['response_triggered']}</td>"
            f"<td style='font-size:0.7rem'>{a['timestamp'][:19]}</td>"
            f"</tr>"
        )

    if not alert_rows:
        alert_rows = "<tr><td colspan='7' style='text-align:center;color:#666'>No alerts yet</td></tr>"

    auto_status = "ON" if auto_enabled else "OFF"
    auto_color = "#22c55e" if auto_enabled else "#6b7280"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PipelineGuard Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 1.5rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; color: #38bdf8; }}
  .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .stat {{ background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; flex: 1; min-width: 140px; }}
  .stat .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }}
  .status-badge {{ display: inline-block; background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; border-radius: 6px; padding: 0.4rem 1rem; font-weight: 700; font-size: 1.1rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; padding: 0.6rem 0.8rem; text-align: left; }}
  td {{ padding: 0.55rem 0.8rem; border-top: 1px solid #0f172a; font-size: 0.85rem; }}
  tr:hover td {{ background: #263248; }}
  h2 {{ font-size: 1rem; color: #94a3b8; margin-bottom: 0.6rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .actions {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .btn {{ padding: 0.5rem 1.2rem; border-radius: 6px; border: none; cursor: pointer; font-size: 0.85rem; font-weight: 600; }}
  .btn-normal {{ background: #22c55e; color: #000; }}
  .btn-poison {{ background: #f59e0b; color: #000; }}
  .btn-adv {{ background: #ef4444; color: #fff; }}
  .btn-clear {{ background: #475569; color: #fff; }}
  .auto-badge {{ background: {auto_color}22; color: {auto_color}; border: 1px solid {auto_color}; border-radius: 6px; padding: 0.25rem 0.75rem; font-size: 0.8rem; font-weight: 700; }}
  .footer {{ color: #475569; font-size: 0.75rem; margin-top: 1rem; }}
</style>
</head>
<body>
<h1>🛡 PipelineGuard</h1>
<p class="subtitle">Healthcare AI Pipeline Security Monitor &nbsp;·&nbsp; Auto-refreshes every 5s &nbsp;·&nbsp; Auto-run: <span class="auto-badge">{auto_status}</span></p>

<div class="stats">
  <div class="stat"><div class="label">Total Runs</div><div class="value" style="color:#38bdf8">{total_runs}</div></div>
  <div class="stat"><div class="label">Total Alerts</div><div class="value" style="color:#f59e0b">{total_alerts}</div></div>
  <div class="stat"><div class="label">Quarantined</div><div class="value" style="color:#ef4444">{quarantine_count}</div></div>
  <div class="stat"><div class="label">Sensitivity</div><div class="value" style="color:#a78bfa;font-size:1.3rem">{SENSITIVITY.upper()}</div></div>
</div>

<div class="status-badge">PIPELINE STATUS: {status_text}</div>

<div class="actions">
  <button class="btn btn-normal" onclick="triggerRun('normal')">▶ Run Normal</button>
  <button class="btn btn-poison" onclick="triggerRun('poisoned')">☣ Run Poisoned</button>
  <button class="btn btn-adv" onclick="triggerRun('adversarial')">⚡ Run Adversarial</button>
  <button class="btn btn-clear" onclick="clearAlerts()">✕ Clear Alerts</button>
</div>

<h2>Recent Alerts (last 10)</h2>
<table>
  <thead><tr><th>ID</th><th>Stage</th><th>Threat Type</th><th>Severity</th><th>Score</th><th>Response</th><th>Timestamp</th></tr></thead>
  <tbody>{alert_rows}</tbody>
</table>

<h2>Stage Telemetry</h2>
<table>
  <thead><tr><th>Stage</th><th>Records</th><th>Key Metrics</th></tr></thead>
  <tbody>{stage_rows}</tbody>
</table>

<p class="footer">Last render: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC &nbsp;·&nbsp; Model: {OLLAMA_MODEL} &nbsp;·&nbsp; Interval: {RUN_INTERVAL}s</p>

<script>
async function triggerRun(mode) {{
  const resp = await fetch('/run', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{mode}})
  }});
  const data = await resp.json();
  alert('Run complete: ' + JSON.stringify(data.alerts_triggered?.length ?? 0) + ' alert(s) triggered');
  location.reload();
}}
async function clearAlerts() {{
  await fetch('/alerts', {{method: 'DELETE'}});
  location.reload();
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard — auto-refreshes every 5 seconds."""
    return _build_dashboard(
        app.state.engine,
        app.state.telemetry,
        app.state.runner,
        app.state.auto_run_enabled,
    )


@app.post("/run")
async def trigger_run(req: RunRequest):
    """Trigger a single pipeline run."""
    if req.mode not in ("normal", "poisoned", "adversarial"):
        raise HTTPException(status_code=400, detail="mode must be 'normal', 'poisoned', or 'adversarial'")
    result = app.state.runner.run(attack_mode=req.mode)
    return JSONResponse(content=result)


@app.post("/run/auto")
async def toggle_auto_run(req: AutoRunRequest):
    """Enable or disable automatic scheduled pipeline runs."""
    scheduler: BackgroundScheduler = app.state.scheduler
    if req.enabled and not app.state.auto_run_enabled:
        if not scheduler.get_job("auto_run"):
            scheduler.add_job(
                lambda: app.state.runner.run(attack_mode="normal"),
                "interval",
                seconds=RUN_INTERVAL,
                id="auto_run",
            )
        app.state.auto_run_enabled = True
    elif not req.enabled and app.state.auto_run_enabled:
        job = scheduler.get_job("auto_run")
        if job:
            job.remove()
        app.state.auto_run_enabled = False
    return {"auto_run_enabled": app.state.auto_run_enabled, "interval_seconds": RUN_INTERVAL}


@app.get("/alerts")
async def get_alerts():
    """Return all alerts, most recent first, up to 100."""
    return app.state.engine.get_alerts(100)


@app.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Return a single alert by ID."""
    for alert in app.state.engine.alerts:
        if alert.id == alert_id:
            return alert.to_dict()
    raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found")


@app.get("/telemetry/{stage}")
async def get_telemetry(stage: str):
    """Return the last 20 telemetry readings for a stage."""
    readings = app.state.telemetry.get_recent(stage, n=20)
    if not readings:
        raise HTTPException(status_code=404, detail=f"No telemetry for stage {stage!r}")
    return [
        {
            "stage": r.stage,
            "timestamp": r.timestamp.isoformat(),
            "batch_id": r.batch_id,
            "record_count": r.record_count,
            "metrics": r.metrics,
        }
        for r in readings
    ]


@app.get("/health")
async def health():
    """Basic health check including Ollama connectivity."""
    ollama_ok = app.state.ollama.is_available()
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "model": OLLAMA_MODEL,
        "sensitivity": SENSITIVITY,
        "auto_run": app.state.auto_run_enabled,
    }


@app.delete("/alerts")
async def clear_alerts():
    """Clear all alerts and quarantine state (for demo resets)."""
    app.state.engine.clear_alerts()
    return {"cleared": True}
