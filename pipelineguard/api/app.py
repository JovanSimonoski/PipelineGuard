"""FastAPI application: REST endpoints + live HTML dashboard."""

import asyncio
import os
import threading
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
WARMUP_RUNS = 5


# ---------------------------------------------------------------------------
# Lifespan — warmup runs in a background thread so the server starts instantly
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
    app.state.warmup_done = False

    def _background_init():
        """Pull model and seed baseline without blocking FastAPI startup."""
        try:
            ollama.ensure_model_pulled()
            console.print(
                f"\n[PipelineGuard] Warming up baseline with {WARMUP_RUNS} normal runs...",
                style="bold cyan",
            )
            for i in range(WARMUP_RUNS):
                runner.run(attack_mode="normal")
                if (i + 1) % 5 == 0:
                    console.print(f"[PipelineGuard] Warmup {i + 1}/{WARMUP_RUNS}", style="cyan")
            console.print("[PipelineGuard] Warmup complete.", style="bold green")
        except Exception as exc:
            console.print(f"[PipelineGuard] Warmup error: {exc}", style="bold red")
        finally:
            app.state.warmup_done = True

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

    threading.Thread(target=_background_init, daemon=True).start()

    yield

    scheduler = app.state.scheduler
    if scheduler and scheduler.running:
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

_SKEL = "<div class='skel'></div>"
_SKEL_ROW_7 = f"<tr><td colspan='7'>{_SKEL}</td></tr>"
_SKEL_ROW_3 = f"<tr><td colspan='3'>{_SKEL}</td></tr>"


def _build_dashboard(
    engine: AnomalyEngine,
    telemetry: TelemetryStore,
    runner: PipelineRunner,
    auto_enabled: bool,
    warmup_done: bool,
) -> str:
    alerts = engine.get_alerts(10)
    total_alerts = engine.alert_count
    quarantine_count = engine.quarantine_count
    total_runs = runner.run_count

    if not alerts:
        status_color, status_text = "#22c55e", "HEALTHY"
    elif any(a["severity"] in ("critical", "high") for a in alerts[:3]):
        status_color, status_text = "#ef4444", "CRITICAL"
    else:
        status_color, status_text = "#f59e0b", "WARNING"

    # Telemetry rows
    stage_rows = ""
    for stage in ["collection", "preprocessing", "inference", "postprocessing", "output_delivery"]:
        latest = telemetry.get_latest(stage)
        if latest:
            top_metrics = [(k, v) for k, v in latest.metrics.items() if isinstance(v, (int, float))][:4]
            metrics_str = " | ".join(f"{k}={v}" for k, v in top_metrics)
            stage_rows += (
                f"<tr><td>{stage}</td><td>{latest.record_count}</td>"
                f"<td class='metric-cell'>{metrics_str}</td></tr>"
            )
        elif not warmup_done:
            stage_rows += _SKEL_ROW_3
        else:
            stage_rows += f"<tr><td>{stage}</td><td>&#x2014;</td><td class='muted'>no data yet</td></tr>"

    # Alert rows
    severity_colors = {"low": "#6b7280", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}
    alert_rows = ""
    for a in alerts:
        color = severity_colors.get(a["severity"], "#6b7280")
        alert_rows += (
            f"<tr>"
            f"<td>{a['id']}</td><td>{a['stage']}</td><td>{a['threat_type']}</td>"
            f"<td style='color:{color};font-weight:bold'>{a['severity'].upper()}</td>"
            f"<td>{a['score']}</td>"
            f"<td class='metric-cell'>{a['response_triggered']}</td>"
            f"<td class='ts-cell'>{a['timestamp'][:19]}</td>"
            f"</tr>"
        )

    if not alert_rows:
        if not warmup_done:
            alert_rows = _SKEL_ROW_7 * 3
        else:
            alert_rows = "<tr><td colspan='7' class='empty-cell'>No alerts yet</td></tr>"

    auto_status = "ON" if auto_enabled else "OFF"
    auto_color = "#22c55e" if auto_enabled else "#6b7280"
    warmup_banner = (
        "<div class='warmup-banner'><div class='spinner'></div>"
        "<span>Warming up &#x2014; pulling model and seeding baseline. Dashboard populates automatically.</span></div>"
        if not warmup_done else ""
    )
    render_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script>document.documentElement.className = localStorage.getItem('pg-theme') || 'dark';</script>
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PipelineGuard Dashboard</title>
<style>
  html.dark {{
    --bg-page: #0f172a; --bg-card: #1e293b; --bg-header: #0f172a;
    --text-primary: #e2e8f0; --text-secondary: #94a3b8; --text-muted: #64748b; --text-disabled: #475569;
    --border-color: #0f172a; --row-hover: #263248; --skel-base: #1e293b; --skel-shine: #263248;
  }}
  html.light {{
    --bg-page: #f1f5f9; --bg-card: #ffffff; --bg-header: #f8fafc;
    --text-primary: #1e293b; --text-secondary: #475569; --text-muted: #64748b; --text-disabled: #94a3b8;
    --border-color: #e2e8f0; --row-hover: #f1f5f9; --skel-base: #e2e8f0; --skel-shine: #f8fafc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg-page); color: var(--text-primary); padding: 1.5rem; }}
  .header-bar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }}
  h1 {{ font-size: 1.6rem; color: #38bdf8; }}
  .subtitle {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .stat {{ background: var(--bg-card); border-radius: 8px; padding: 1rem 1.5rem; flex: 1; min-width: 140px; }}
  .stat .label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }}
  .status-badge {{ display: inline-block; background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; border-radius: 6px; padding: 0.4rem 1rem; font-weight: 700; font-size: 1.1rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--bg-card); border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; }}
  th {{ background: var(--bg-header); color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; padding: 0.6rem 0.8rem; text-align: left; }}
  td {{ padding: 0.55rem 0.8rem; border-top: 1px solid var(--border-color); font-size: 0.85rem; }}
  .metric-cell {{ font-size: 0.75rem; }}
  .ts-cell {{ font-size: 0.7rem; }}
  .muted {{ color: var(--text-muted); }}
  .empty-cell {{ text-align: center; color: var(--text-muted); }}
  tr:hover td {{ background: var(--row-hover); }}
  h2 {{ font-size: 1rem; color: var(--text-secondary); margin-bottom: 0.6rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .actions {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem; align-items: center; }}
  .btn {{ padding: 0.5rem 1.2rem; border-radius: 6px; border: none; cursor: pointer; font-size: 0.85rem; font-weight: 600; }}
  .btn-normal {{ background: #22c55e; color: #000; }}
  .btn-poison {{ background: #f59e0b; color: #000; }}
  .btn-adv {{ background: #ef4444; color: #fff; }}
  .btn-clear {{ background: #475569; color: #fff; }}
  .btn-theme {{ background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--text-muted); border-radius: 6px; padding: 0.4rem 0.75rem; cursor: pointer; font-size: 1rem; line-height: 1; }}
  .auto-badge {{ background: {auto_color}22; color: {auto_color}; border: 1px solid {auto_color}; border-radius: 6px; padding: 0.25rem 0.75rem; font-size: 0.8rem; font-weight: 700; }}
  .footer {{ color: var(--text-disabled); font-size: 0.75rem; margin-top: 1rem; }}
  .warmup-banner {{ display: flex; align-items: center; gap: 0.75rem; background: var(--bg-card); border: 1px solid #38bdf8; border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; color: #38bdf8; font-size: 0.9rem; }}
  .spinner {{ width: 16px; height: 16px; border: 2px solid #38bdf8; border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .skel {{ height: 0.85rem; border-radius: 4px; background: linear-gradient(90deg, var(--skel-base) 25%, var(--skel-shine) 50%, var(--skel-base) 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; }}
  @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
  #toast {{ position: fixed; bottom: 1.5rem; right: 1.5rem; padding: 0.75rem 1.25rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600; z-index: 1000; opacity: 0; pointer-events: none; transition: opacity 0.3s; }}
  #toast.show {{ opacity: 1; pointer-events: auto; }}
  #toast.success {{ background: #22c55e; color: #000; }}
  #toast.error {{ background: #ef4444; color: #fff; }}
</style>
</head>
<body>
<div class="header-bar">
  <h1>&#x1F6E1; PipelineGuard</h1>
  <button id="theme-btn" class="btn-theme" onclick="toggleTheme()" title="Toggle dark/light mode">&#x1F319;</button>
</div>
<p class="subtitle">Healthcare AI Pipeline Security Monitor &nbsp;&middot;&nbsp; Auto-refreshes every 5s &nbsp;&middot;&nbsp; Auto-run: <span class="auto-badge">{auto_status}</span></p>

{warmup_banner}
<div class="stats">
  <div class="stat"><div class="label">Total Runs</div><div class="value" style="color:#38bdf8">{total_runs}</div></div>
  <div class="stat"><div class="label">Total Alerts</div><div class="value" style="color:#f59e0b">{total_alerts}</div></div>
  <div class="stat"><div class="label">Quarantined</div><div class="value" style="color:#ef4444">{quarantine_count}</div></div>
  <div class="stat"><div class="label">Sensitivity</div><div class="value" style="color:#a78bfa;font-size:1.3rem">{SENSITIVITY.upper()}</div></div>
</div>

<div class="status-badge">PIPELINE STATUS: {status_text}</div>

<div class="actions">
  <button class="btn btn-normal" onclick="triggerRun('normal')">&#x25B6; Run Normal</button>
  <button class="btn btn-poison" onclick="triggerRun('poisoned')">&#x2623; Run Poisoned</button>
  <button class="btn btn-adv" onclick="triggerRun('adversarial')">&#x26A1; Run Adversarial</button>
  <button class="btn btn-clear" onclick="clearAlerts()">&#x2715; Clear Alerts</button>
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

<p class="footer">Last render: {render_time} UTC &nbsp;&middot;&nbsp; Model: {OLLAMA_MODEL} &nbsp;&middot;&nbsp; Interval: {RUN_INTERVAL}s</p>
<div id="toast"></div>

<script>
(function() {{
  var cls = localStorage.getItem('pg-theme') || 'dark';
  document.documentElement.className = cls;
  var btn = document.getElementById('theme-btn');
  if (btn) btn.innerHTML = cls === 'dark' ? '&#x1F319;' : '&#x2600;&#xFE0F;';
}})();

function toggleTheme() {{
  var html = document.documentElement;
  var next = html.classList.contains('light') ? 'dark' : 'light';
  html.className = next;
  localStorage.setItem('pg-theme', next);
  document.getElementById('theme-btn').innerHTML = next === 'dark' ? '&#x1F319;' : '&#x2600;&#xFE0F;';
}}

function showToast(msg, type) {{
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + type;
  setTimeout(function() {{ t.className = ''; }}, 3000);
}}

async function triggerRun(mode) {{
  try {{
    const resp = await fetch('/run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{mode}})
    }});
    if (!resp.ok) {{
      const err = await resp.json().catch(function() {{ return {{detail: 'Unknown error'}}; }});
      throw new Error(err.detail || 'Request failed');
    }}
    const data = await resp.json();
    const n = (data.alerts_triggered || []).length;
    showToast('Run complete — ' + n + ' alert(s)', 'success');
    setTimeout(function() {{ location.reload(); }}, 800);
  }} catch (err) {{
    showToast('Error: ' + err.message, 'error');
  }}
}}

async function clearAlerts() {{
  try {{
    const resp = await fetch('/alerts', {{method: 'DELETE'}});
    if (!resp.ok) throw new Error('Failed to clear alerts');
    showToast('Alerts cleared', 'success');
    setTimeout(function() {{ location.reload(); }}, 800);
  }} catch (err) {{
    showToast('Error: ' + err.message, 'error');
  }}
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
        app.state.warmup_done,
    )


@app.post("/run")
async def trigger_run(req: RunRequest):
    """Trigger a single pipeline run (offloaded to thread pool to avoid blocking the event loop)."""
    if req.mode not in ("normal", "poisoned", "adversarial"):
        raise HTTPException(status_code=400, detail="mode must be 'normal', 'poisoned', or 'adversarial'")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: app.state.runner.run(attack_mode=req.mode))
    return JSONResponse(content=result)


@app.post("/run/auto")
async def toggle_auto_run(req: AutoRunRequest):
    """Enable or disable automatic scheduled pipeline runs."""
    scheduler: BackgroundScheduler = app.state.scheduler
    if scheduler is None:
        raise HTTPException(status_code=503, detail="System is still warming up")
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
    """Basic health check including Ollama connectivity and warmup status."""
    ollama_ok = app.state.ollama.is_available()
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "model": OLLAMA_MODEL,
        "sensitivity": SENSITIVITY,
        "auto_run": app.state.auto_run_enabled,
        "warmup_done": app.state.warmup_done,
    }


@app.delete("/alerts")
async def clear_alerts():
    """Clear all alerts and quarantine state (for demo resets)."""
    app.state.engine.clear_alerts()
    return {"cleared": True}
