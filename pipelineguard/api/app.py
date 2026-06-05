# -*- coding: utf-8 -*-
"""FastAPI application: REST endpoints + live HTML dashboard."""

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np

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

SENSITIVITY  = os.getenv("ANOMALY_SENSITIVITY",   "medium")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",            "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",           "tinyllama")
RUN_INTERVAL = int(os.getenv("PIPELINE_RUN_INTERVAL", "20"))
WARMUP_RUNS  = 2

_SKEL      = "<div class='skel'></div>"
_SKEL_ROW7 = "<tr><td colspan='7'>" + _SKEL + "</td></tr>"
_SKEL_ROW3 = "<tr><td colspan='3'>" + _SKEL + "</td></tr>"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    telemetry = TelemetryStore()
    engine    = AnomalyEngine(sensitivity=SENSITIVITY)
    ollama    = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL)
    runner    = PipelineRunner(engine=engine, telemetry=telemetry, ollama_client=ollama)

    app.state.telemetry        = telemetry
    app.state.engine           = engine
    app.state.ollama           = ollama
    app.state.runner           = runner
    app.state.scheduler        = None
    app.state.auto_run_enabled = False
    app.state.warmup_done      = False
    app.state.start_time       = datetime.now(timezone.utc)

    def _background_init():
        """Pull model and seed baseline without blocking FastAPI startup."""
        try:
            ollama.ensure_model_pulled()
            console.print(
                f"\n[PipelineGuard] Warming up with {WARMUP_RUNS} baseline runs...",
                style="bold cyan",
            )
            for i in range(WARMUP_RUNS):
                runner.run(attack_mode="normal")
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
        app.state.scheduler        = scheduler
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
# Dashboard helpers
# ---------------------------------------------------------------------------

def _activity_feed_html(engine: AnomalyEngine, telemetry: TelemetryStore, warmup_done: bool) -> str:
    alerts_raw      = engine.get_alerts(30)
    alert_batch_ids = {a.get("batch_id", "") for a in alerts_raw}

    ICONS  = {"critical": "&#x1F534;", "high": "&#x1F7E0;", "medium": "&#x26A0;", "low": "&#x1F535;"}
    LABELS = {"critical": "Critical threat", "high": "High anomaly", "medium": "Anomaly detected", "low": "Low-severity event"}

    events = []
    for a in alerts_raw:
        sev    = a["severity"]
        threat = a["threat_type"].replace("_", " ").title()
        stage  = a["stage"].replace("_", " ").title()
        events.append({
            "ts":     a["timestamp"][:19].replace("T", " "),
            "icon":   ICONS.get(sev, "&#x26A0;"),
            "label":  LABELS.get(sev, "Alert") + " &#x2014; " + threat,
            "detail": stage + " &middot; score " + str(a["score"]) + " &middot; " + a["response_triggered"],
            "cls":    "fi-" + sev,
        })

    for r in telemetry.get_recent("collection", n=20):
        if r.batch_id not in alert_batch_ids:
            events.append({
                "ts":     r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "icon":   "&#x2705;",
                "label":  "Clean batch &mdash; " + str(r.record_count) + " records",
                "detail": "All 5 stages cleared &middot; no threats",
                "cls":    "fi-ok",
            })

    events.sort(key=lambda e: e["ts"], reverse=True)
    events = events[:30]

    # Collapse consecutive clean runs into one entry with a repeat count
    collapsed = []
    for ev in events:
        if ev["cls"] == "fi-ok" and collapsed and collapsed[-1]["cls"] == "fi-ok":
            collapsed[-1]["count"] = collapsed[-1].get("count", 1) + 1
        else:
            ev["count"] = 1
            collapsed.append(ev)
    events = collapsed[:12]

    if not events:
        if not warmup_done:
            return "<div class='feed-empty'><div class='spinner'></div><span>Seeding baseline&hellip;</span></div>"
        return "<div class='feed-empty'>No activity yet</div>"

    html = ""
    for ev in events:
        count = ev.get("count", 1)
        repeat = (" <span class='rep'>x" + str(count) + "</span>") if count > 1 else ""
        html += (
            "<div class='feed-item " + ev["cls"] + "'>"
            "<div class='fi-icon'>" + ev["icon"] + "</div>"
            "<div class='fi-body'>"
            "<div class='fi-label'>" + ev["label"] + repeat + "</div>"
            "<div class='fi-detail'>" + ev["detail"] + "</div>"
            "<div class='fi-ts'>" + ev["ts"] + "</div>"
            "</div></div>"
        )
    return html


# ---------------------------------------------------------------------------
# Dashboard helpers – explainability panels
# ---------------------------------------------------------------------------

_STAGE_ICONS = {
    "collection":      "&#x1F4E5;",
    "preprocessing":   "&#x2699;",
    "inference":       "&#x1F9E0;",
    "postprocessing":  "&#x1F4CA;",
    "output_delivery": "&#x1F4E4;",
}

def _pipeline_flow_html(runner, engine: AnomalyEngine, warmup_done: bool) -> str:
    """Horizontal stage-flow strip with per-stage status."""
    last = runner.last_run or {}
    stage_results = last.get("stage_results", {})
    attack_mode = last.get("attack_mode", "normal")
    MODE_COLORS = {"normal": "#22c55e", "poisoned": "#f59e0b", "adversarial": "#ef4444"}
    mode_color = MODE_COLORS.get(attack_mode, "#94a3b8")

    STATUS_CFG = {
        "ok":          ("#22c55e", "OK",         "Clean"),
        "alert":       ("#ef4444", "ALERT",       "Threat detected"),
        "quarantined": ("#f59e0b", "QUARANTINED", "Pipeline stopped"),
        "skipped":     ("#374151", "SKIPPED",     "Not reached"),
    }

    stages = ["collection", "preprocessing", "inference", "postprocessing", "output_delivery"]
    stage_labels = ["Collection", "Preprocessing", "AI Inference", "Postprocessing", "Delivery"]

    html = "<div class='pipeline-flow'>"
    if last:
        mode_badge = (
            f"<span class='run-mode-badge' style='background:{mode_color}22;"
            f"color:{mode_color};border-color:{mode_color}55'>"
            f"Last run: {attack_mode.upper()}"
            f" &nbsp;&#x231A;&nbsp; {last.get('duration_ms', 0)} ms"
            f" &nbsp;&#x2713;&nbsp; {last.get('stages_completed', 0)}/5 stages</span>"
        )
        html += f"<div class='flow-meta'>{mode_badge}</div>"

    html += "<div class='flow-stages'>"
    for i, (sn, label) in enumerate(zip(stages, stage_labels)):
        status = stage_results.get(sn, "skipped") if last else ("seeding" if not warmup_done else "pending")
        if status == "seeding" or status == "pending":
            clr, tag, tip = "#94a3b8", "–", "Waiting"
        else:
            clr, tag, tip = STATUS_CFG.get(status, STATUS_CFG["skipped"])

        icon = _STAGE_ICONS.get(sn, "&#x25A1;")
        arrow = "<div class='flow-arrow'>&#x203A;</div>" if i < len(stages) - 1 else ""
        html += (
            f"<div class='flow-stage' title='{tip}'>"
            f"<div class='fs-icon' style='background:{clr}22;border-color:{clr}44'>{icon}</div>"
            f"<div class='fs-name'>{label}</div>"
            f"<div class='fs-tag' style='color:{clr}'>{tag}</div>"
            f"</div>{arrow}"
        )
    html += "</div></div>"
    return html


def _baseline_panel_html(engine: AnomalyEngine, telemetry: TelemetryStore) -> str:
    """Per-stage baseline maturity + current metric vs baseline."""
    snap = engine.baseline.snapshot()
    stages = ["collection", "preprocessing", "inference", "postprocessing", "output_delivery"]
    WINDOW = 30

    html = "<div class='baseline-grid'>"
    for stage in stages:
        stage_metrics = snap.get(stage, {})
        latest = telemetry.get_latest(stage)
        icon = _STAGE_ICONS.get(stage, "")
        label = stage.replace("_", " ").title()

        # overall sample count = min across metrics (or 0)
        counts = [len(v) for v in stage_metrics.values() if v]
        sample_n = min(counts) if counts else 0
        pct = min(int(sample_n / WINDOW * 100), 100)
        established = sample_n >= 10
        bar_color = "#22c55e" if established else "#f59e0b"

        html += (
            f"<div class='bl-card'>"
            f"<div class='bl-header'>"
            f"<span class='bl-icon'>{icon}</span>"
            f"<span class='bl-title'>{label}</span>"
            f"<span class='bl-status' style='color:{bar_color}'>"
            f"{'&#x2713; Established' if established else '&#x23F3; Seeding'}"
            f"</span></div>"
            f"<div class='bl-bar-wrap'>"
            f"<div class='bl-bar' style='width:{pct}%;background:{bar_color}'></div>"
            f"</div>"
            f"<div class='bl-bar-lbl'>{sample_n}/{WINDOW} samples</div>"
        )

        # per-metric rows
        if latest and stage_metrics:
            html += "<table class='bl-metrics'><thead><tr><th>Metric</th><th>Current</th><th>Baseline μ</th><th>Δσ (Z)</th></tr></thead><tbody>"
            for metric, vals in list(stage_metrics.items())[:5]:
                cur = latest.metrics.get(metric)
                if cur is None or not isinstance(cur, (int, float)):
                    continue
                arr = [float(x) for x in vals]
                mean = float(np.mean(arr)) if arr else None
                std  = float(np.std(arr))  if arr else None
                if mean is not None and std is not None and std > 1e-6:
                    z = (cur - mean) / std
                    z_str = f"{z:+.2f}"
                    z_color = "#ef4444" if abs(z) > 2.5 else ("#f59e0b" if abs(z) > 1.5 else "#22c55e")
                else:
                    z_str, z_color = "–", "var(--muted)"
                mean_str = f"{mean:.3f}" if mean is not None else "–"
                cur_str  = f"{cur:.3f}" if isinstance(cur, float) else str(cur)
                html += (
                    f"<tr><td class='bl-m-name'>{metric}</td>"
                    f"<td>{cur_str}</td>"
                    f"<td class='muted'>{mean_str}</td>"
                    f"<td style='color:{z_color};font-weight:700'>{z_str}</td></tr>"
                )
            html += "</tbody></table>"
        html += "</div>"
    html += "</div>"
    return html


def _alert_details_js_data(alerts: list[dict]) -> str:
    """Emit alert detail data as a JS object so rows can be expanded inline."""
    detail_map = {}
    for a in alerts:
        detail_map[a["id"]] = a.get("details", [])
    return "var ALERT_DETAILS = " + json.dumps(detail_map) + ";"


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

def _build_dashboard(
    engine:      AnomalyEngine,
    telemetry:   TelemetryStore,
    runner:      PipelineRunner,
    auto_enabled: bool,
    warmup_done:  bool,
    start_time:   datetime,
) -> str:

    # --- stats --------------------------------------------------------------
    alerts         = engine.get_alerts(10)
    total_alerts   = engine.alert_count
    quarantined    = engine.quarantine_count
    total_runs     = runner.run_count

    # --- pipeline status ----------------------------------------------------
    if not alerts:
        sc, st, sh = "#22c55e", "HEALTHY",  "No threats in recent activity"
    elif any(a["severity"] in ("critical", "high") for a in alerts[:3]):
        sc, st, sh = "#ef4444", "CRITICAL", "Active threat &mdash; quarantine triggered"
    else:
        sc, st, sh = "#f59e0b", "WARNING",  "Anomalies present &mdash; elevated monitoring"

    # --- telemetry table rows -----------------------------------------------
    stage_rows = ""
    for stage in ["collection", "preprocessing", "inference", "postprocessing", "output_delivery"]:
        latest = telemetry.get_latest(stage)
        if latest:
            top = [(k, v) for k, v in latest.metrics.items() if isinstance(v, (int, float))][:4]
            mstr = " | ".join(k + "=" + str(v) for k, v in top)
            stage_rows += (
                "<tr><td>" + stage + "</td><td>" + str(latest.record_count) + "</td>"
                "<td class='mc'>" + mstr + "</td></tr>"
            )
        elif not warmup_done:
            stage_rows += _SKEL_ROW3
        else:
            stage_rows += "<tr><td>" + stage + "</td><td>&mdash;</td><td class='muted'>no data</td></tr>"

    # --- alert table rows ---------------------------------------------------
    SEV_CLR = {"low": "#6b7280", "medium": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}
    alert_rows = ""
    for a in alerts:
        clr = SEV_CLR.get(a["severity"], "#6b7280")
        detail_id = "det-" + a["id"]
        alert_rows += (
            "<tr class='expandable' onclick=\"toggleDetail('" + a["id"] + "')\">"
            "<td><span class='expand-icon'>&#x25B6;</span>" + a["id"] + "</td>"
            "<td>" + a["stage"] + "</td>"
            "<td>" + a["threat_type"] + "</td>"
            "<td style='color:" + clr + ";font-weight:700'>" + a["severity"].upper() + "</td>"
            "<td>" + str(a["score"]) + "</td>"
            "<td class='mc'>" + a["response_triggered"] + "</td>"
            "<td class='tc'>" + a["timestamp"][:19] + "</td>"
            "</tr>"
            "<tr class='alert-detail-row' id='" + detail_id + "' style='display:none'>"
            "<td colspan='7'><div class='alert-detail-wrap' id='adw-" + a["id"] + "'></div></td>"
            "</tr>"
        )
    if not alert_rows:
        alert_rows = (
            _SKEL_ROW7 * 2 if not warmup_done
            else "<tr><td colspan='7' class='empty'>No alerts &mdash; pipeline running clean</td></tr>"
        )

    # --- new explainability panels -----------------------------------------
    flow_html      = _pipeline_flow_html(runner, engine, warmup_done)
    baseline_html  = _baseline_panel_html(engine, telemetry)
    alert_detail_js = _alert_details_js_data(engine.get_alerts(10))

    # --- misc ---------------------------------------------------------------
    feed_html    = _activity_feed_html(engine, telemetry, warmup_done)
    feed_count   = len(alerts) + len(telemetry.get_recent("collection", n=20))
    auto_color   = "#22c55e" if auto_enabled else "#6b7280"
    auto_lbl     = "ON" if auto_enabled else "OFF"

    uptime_sec   = int((datetime.now(timezone.utc) - start_time).total_seconds())
    if uptime_sec < 60:
        uptime_str = str(uptime_sec) + "s"
    elif uptime_sec < 3600:
        uptime_str = str(uptime_sec // 60) + "m " + str(uptime_sec % 60) + "s"
    else:
        uptime_str = str(uptime_sec // 3600) + "h " + str((uptime_sec % 3600) // 60) + "m"

    if not warmup_done:
        sys_status, detect_mode, dot_cls = "Warming Up", "Seeding", "dot-warm"
    elif auto_enabled:
        sys_status, detect_mode, dot_cls = "Online",     "Active",  "dot-on"
    else:
        sys_status, detect_mode, dot_cls = "Online",     "Passive", "dot-on"

    warmup_banner = (
        "<div class='warmup-banner'><div class='spinner'></div>"
        "<span>Warming up &mdash; pulling model and seeding detection baseline.</span></div>"
        if not warmup_done else ""
    )
    render_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # f-string only for the parts that need Python values embedded
    status_badge_css = (
        "background:" + sc + "1a;color:" + sc + ";border:1px solid " + sc + "55;"
    )

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script>document.documentElement.className = localStorage.getItem('pg-theme') || 'dark';</script>
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PipelineGuard</title>
<style>
  html.dark {
    --bg:     #0b1120; --card:   #111827; --raised: #1a2436; --hdr: #0d1525;
    --txt:    #f1f5f9; --sub:    #94a3b8; --muted:  #64748b; --dim: #374151;
    --border: #1e2d42; --hover:  #172032; --sb: #030709;
    --sk1: #111827; --sk2: #1a2436;
  }
  html.light {
    --bg:     #f0f4f8; --card:   #ffffff; --raised: #f8fafc; --hdr: #f1f5f9;
    --txt:    #0f172a; --sub:    #475569; --muted:  #64748b; --dim: #94a3b8;
    --border: #e2e8f0; --hover:  #f8fafc; --sb: #e8ecf0;
    --sk1: #e2e8f0; --sk2: #f8fafc;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--txt); padding:1.25rem 1.5rem 3.2rem; }

  /* header */
  .header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.25rem; gap:1rem; }
  .brand h1 { font-size:1.6rem; color:#38bdf8; font-weight:800; letter-spacing:-0.02em; }
  .brand .tagline { font-size:0.8rem; color:var(--muted); margin-top:0.2rem; }
  .brand .steps { font-size:0.72rem; color:var(--dim); margin-top:0.3rem; letter-spacing:0.01em; }
  .btn-theme { background:var(--card); color:var(--txt); border:1px solid var(--border); border-radius:6px; padding:0.35rem 0.65rem; cursor:pointer; font-size:0.95rem; }

  /* warmup */
  .warmup-banner { display:flex; align-items:center; gap:0.65rem; background:var(--card); border:1px solid #38bdf844; border-radius:8px; padding:0.65rem 1rem; margin-bottom:1.25rem; color:#38bdf8; font-size:0.82rem; }
  .spinner { width:14px; height:14px; border:2px solid #38bdf8; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; flex-shrink:0; }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* main grid */
  .grid { display:grid; grid-template-columns:1fr 280px; gap:1.25rem; align-items:start; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } }

  /* stats */
  .stats { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1.1rem; }
  .stat { background:var(--card); border-radius:8px; padding:0.85rem 1rem; flex:1; min-width:115px; }
  .stat .lbl { font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; font-weight:700; }
  .stat .val { font-size:1.9rem; font-weight:800; margin:0.15rem 0 0.2rem; line-height:1; }
  .stat .hint { font-size:0.62rem; color:var(--dim); }

  /* status */
  .status-wrap { margin-bottom:1rem; }
  .status-badge { display:inline-block; border-radius:6px; padding:0.3rem 0.8rem; font-weight:800; font-size:0.9rem; letter-spacing:0.03em; }
  .status-hint { font-size:0.7rem; color:var(--muted); margin-top:0.3rem; }

  /* actions */
  .actions { display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.85rem; align-items:center; }
  .btn { padding:0.4rem 0.9rem; border-radius:6px; border:none; cursor:pointer; font-size:0.78rem; font-weight:700; }
  .btn:hover { opacity:0.85; }
  .btn-n { background:#22c55e; color:#000; }
  .btn-p { background:#f59e0b; color:#000; }
  .btn-a { background:#ef4444; color:#fff; }
  .btn-c { background:#374151; color:#9ca3af; }
  .auto-badge { font-size:0.7rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:20px; border:1px solid; }

  /* section labels */
  .sec-lbl { font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.09em; font-weight:700; margin:1rem 0 0.4rem; }

  /* tables */
  table { width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; margin-bottom:0.5rem; }
  th { background:var(--hdr); color:var(--sub); font-size:0.62rem; text-transform:uppercase; letter-spacing:0.07em; padding:0.5rem 0.7rem; text-align:left; font-weight:700; }
  td { padding:0.45rem 0.7rem; border-top:1px solid var(--border); font-size:0.8rem; }
  .mc { font-size:0.68rem; color:var(--sub); }
  .tc { font-size:0.65rem; color:var(--sub); }
  .muted { color:var(--muted); }
  .empty { text-align:center; color:var(--muted); padding:1rem; }
  tr:hover td { background:var(--hover); }

  /* feed */
  .feed { background:var(--card); border-radius:8px; overflow:hidden; position:sticky; top:1rem; }
  .feed-hdr { display:flex; justify-content:space-between; align-items:center; padding:0.55rem 0.8rem; background:var(--hdr); border-bottom:1px solid var(--border); }
  .feed-title { font-size:0.62rem; text-transform:uppercase; letter-spacing:0.09em; font-weight:700; color:var(--sub); }
  .feed-ct { font-size:0.6rem; background:var(--raised); color:var(--muted); border-radius:20px; padding:0.12rem 0.45rem; }
  .feed-body { max-height:72vh; overflow-y:auto; }
  .feed-item { display:flex; gap:0.5rem; padding:0.55rem 0.8rem; border-top:1px solid var(--border); border-left:3px solid transparent; }
  .feed-item:first-child { border-top:none; }
  .feed-item:hover { background:var(--hover); }
  .fi-critical { border-left-color:#ef4444; }
  .fi-high     { border-left-color:#f97316; }
  .fi-medium   { border-left-color:#f59e0b; }
  .fi-low      { border-left-color:#6b7280; }
  .fi-icon     { font-size:0.9rem; flex-shrink:0; padding-top:0.05rem; }
  .fi-label    { font-size:0.75rem; font-weight:600; margin-bottom:0.1rem; line-height:1.3; }
  .fi-detail   { font-size:0.63rem; color:var(--muted); margin-bottom:0.15rem; }
  .fi-ts       { font-size:0.58rem; color:var(--dim); }
  .rep         { display:inline-block; background:var(--raised); color:var(--muted); border-radius:10px; padding:0.05rem 0.35rem; font-size:0.58rem; font-weight:700; margin-left:0.3rem; vertical-align:middle; }
  .feed-empty  { padding:1.5rem; text-align:center; color:var(--muted); font-size:0.78rem; display:flex; align-items:center; justify-content:center; gap:0.5rem; }

  /* status bar */
  .sb { position:fixed; bottom:0; left:0; right:0; background:var(--sb); border-top:1px solid var(--border); padding:0.38rem 1.5rem; display:flex; gap:1.5rem; align-items:center; font-size:0.68rem; color:var(--sub); z-index:200; flex-wrap:wrap; }
  .sb-it { display:flex; align-items:center; gap:0.3rem; }
  .sb-dot { width:6px; height:6px; border-radius:50%; }
  .dot-on   { background:#22c55e; animation:pulse 2.5s infinite; }
  .dot-warm { background:#f59e0b; animation:pulse 1s infinite; }
  .sb-lbl { color:var(--dim); }
  .sb-val { color:var(--txt); font-weight:600; }
  .sb-r   { margin-left:auto; color:var(--dim); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

  /* skeleton */
  .skel { height:0.78rem; border-radius:3px; background:linear-gradient(90deg,var(--sk1) 25%,var(--sk2) 50%,var(--sk1) 75%); background-size:200% 100%; animation:shimmer 1.5s infinite; }
  @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

  /* toast */
  #toast { position:fixed; bottom:2.8rem; right:1.5rem; padding:0.6rem 1rem; border-radius:7px; font-size:0.8rem; font-weight:600; z-index:1000; opacity:0; pointer-events:none; transition:opacity 0.25s; }
  #toast.show { opacity:1; pointer-events:auto; }
  #toast.ok  { background:#22c55e; color:#000; }
  #toast.err { background:#ef4444; color:#fff; }

  /* pipeline flow */
  .pipeline-flow { background:var(--card); border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; }
  .flow-meta { margin-bottom:0.6rem; }
  .run-mode-badge { display:inline-block; border-radius:20px; border:1px solid; padding:0.2rem 0.7rem; font-size:0.68rem; font-weight:700; }
  .flow-stages { display:flex; align-items:center; gap:0; overflow-x:auto; padding-bottom:0.2rem; }
  .flow-stage { display:flex; flex-direction:column; align-items:center; gap:0.25rem; min-width:90px; }
  .fs-icon { width:40px; height:40px; border-radius:50%; border:2px solid; display:flex; align-items:center; justify-content:center; font-size:1.1rem; }
  .fs-name { font-size:0.6rem; color:var(--sub); text-align:center; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
  .fs-tag  { font-size:0.62rem; font-weight:800; letter-spacing:0.04em; }
  .flow-arrow { color:var(--dim); font-size:1.3rem; padding:0 0.15rem; flex-shrink:0; margin-bottom:1rem; }

  /* alert details expansion */
  .alert-detail-row td { background:var(--raised) !important; padding:0.6rem 0.9rem; }
  .alert-detail-wrap { font-size:0.72rem; }
  .ad-check { display:inline-block; background:var(--card); border:1px solid var(--border); border-radius:5px; padding:0.25rem 0.6rem; margin:0.15rem 0.15rem 0 0; font-size:0.67rem; }
  .ad-key { color:var(--muted); }
  .ad-val { color:var(--txt); font-weight:600; }
  .ad-z   { color:#ef4444; font-weight:700; }
  tr.expandable { cursor:pointer; }
  tr.expandable:hover td { background:var(--hover); }
  .expand-icon { display:inline-block; transition:transform 0.15s; font-size:0.65rem; margin-right:0.3rem; }
  .expanded .expand-icon { transform:rotate(90deg); }

  /* baseline panel */
  .baseline-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:0.75rem; margin-bottom:0.5rem; }
  .bl-card { background:var(--card); border-radius:8px; padding:0.75rem; }
  .bl-header { display:flex; align-items:center; gap:0.4rem; margin-bottom:0.5rem; }
  .bl-icon  { font-size:1rem; }
  .bl-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; flex:1; }
  .bl-status { font-size:0.62rem; font-weight:700; }
  .bl-bar-wrap { background:var(--raised); border-radius:4px; height:5px; margin-bottom:0.25rem; }
  .bl-bar { height:5px; border-radius:4px; transition:width 0.4s; }
  .bl-bar-lbl { font-size:0.58rem; color:var(--muted); margin-bottom:0.5rem; }
  .bl-metrics { width:100%; border-collapse:collapse; font-size:0.65rem; }
  .bl-metrics th { color:var(--dim); font-size:0.58rem; font-weight:700; text-transform:uppercase; padding:0.18rem 0.3rem; border-bottom:1px solid var(--border); }
  .bl-metrics td { padding:0.18rem 0.3rem; border-top:1px solid var(--border); }
  .bl-m-name { color:var(--sub); font-size:0.63rem; }

  /* detector legend */
  .detector-legend { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem; }
  .dl-card { background:var(--card); border-radius:8px; padding:0.65rem 0.85rem; flex:1; min-width:180px; border-left:3px solid; }
  .dl-title { font-size:0.65rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.35rem; }
  .dl-row { font-size:0.63rem; color:var(--sub); margin:0.12rem 0; }
  .dl-row b { color:var(--txt); }
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <h1>&#x1F6E1; PipelineGuard</h1>
    <div class="tagline">Real-time threat detection for healthcare AI inference pipelines</div>
    <div class="steps">Data Collection &#x203A; Preprocessing &#x203A; AI Inference &#x203A; Anomaly Detection &#x203A; Auto-Response</div>
  </div>
  <button id="theme-btn" class="btn-theme" onclick="toggleTheme()" title="Toggle dark/light mode">&#x1F319;</button>
</div>

""" + warmup_banner + """

""" + flow_html + """

<div class="grid">

  <div>
    <div class="stats">
      <div class="stat">
        <div class="lbl">Total Runs</div>
        <div class="val" style="color:#38bdf8">""" + str(total_runs) + """</div>
        <div class="hint">batches processed since startup</div>
      </div>
      <div class="stat">
        <div class="lbl">Alerts</div>
        <div class="val" style="color:#f59e0b">""" + str(total_alerts) + """</div>
        <div class="hint">threats flagged across all stages</div>
      </div>
      <div class="stat">
        <div class="lbl">Quarantined</div>
        <div class="val" style="color:#ef4444">""" + str(quarantined) + """</div>
        <div class="hint">batches blocked before delivery</div>
      </div>
      <div class="stat">
        <div class="lbl">Sensitivity</div>
        <div class="val" style="color:#a78bfa;font-size:1.3rem">""" + SENSITIVITY.upper() + """</div>
        <div class="hint">detection threshold level</div>
      </div>
    </div>

    <div class="status-wrap">
      <div class="status-badge" style=\"""" + status_badge_css + """\">PIPELINE: """ + st + """</div>
      <div class="status-hint">""" + sh + """</div>
    </div>

    <div class="actions">
      <button class="btn btn-n" onclick="triggerRun('normal')"
        title="Process a clean 20-record patient batch">&#x25B6; Run Normal</button>
      <button class="btn btn-p" onclick="triggerRun('poisoned')"
        title="30% of records shifted 2-3 sigma -- simulates data poisoning">&#x2623; Run Poisoned</button>
      <button class="btn btn-a" onclick="triggerRun('adversarial')"
        title="One extreme adversarial record designed to fool the model">&#x26A1; Run Adversarial</button>
      <button class="btn btn-c" onclick="clearAlerts()">&#x2715; Reset</button>
      <span class="auto-badge" style="color:""" + auto_color + """;border-color:""" + auto_color + """44;background:""" + auto_color + """11">Auto """ + auto_lbl + """</span>
    </div>

    <div class="sec-lbl">Recent Alerts</div>
    <table>
      <thead><tr><th>ID</th><th>Stage</th><th>Threat</th><th>Severity</th><th>Score</th><th>Response</th><th>Time (UTC)</th></tr></thead>
      <tbody>""" + alert_rows + """</tbody>
    </table>

    <div class="sec-lbl">Detection Engines</div>
    <div class="detector-legend">
      <div class="dl-card" style="border-color:#f59e0b">
        <div class="dl-title" style="color:#f59e0b">&#x2623; Poisoning Detector</div>
        <div class="dl-row">Watches for <b>statistical drift</b> in any numeric metric</div>
        <div class="dl-row">Method: <b>Z-score</b> against rolling 30-sample baseline</div>
        <div class="dl-row">Fires when Z &gt; <b>""" + str(engine.poisoning.threshold) + """</b> &sigma; (""" + SENSITIVITY + """ sensitivity)</div>
        <div class="dl-row">Stages: <b>all 5 stages</b></div>
      </div>
      <div class="dl-card" style="border-color:#ef4444">
        <div class="dl-title" style="color:#ef4444">&#x26A1; Adversarial Detector</div>
        <div class="dl-row">Watches for <b>model confidence collapse</b> at inference</div>
        <div class="dl-row">Checks: low avg conf &lt; <b>""" + str(engine.adversarial.low_conf_threshold) + """</b>, unknown rate &gt; <b>""" + str(engine.adversarial.unknown_rate_threshold) + """</b></div>
        <div class="dl-row">Also: confidence drop vs baseline (Z &gt; 2.0), high variance (Z &gt; <b>""" + str(engine.adversarial.conf_std_multiplier) + """</b>)</div>
        <div class="dl-row">Stages: <b>inference only</b></div>
      </div>
      <div class="dl-card" style="border-color:#a78bfa">
        <div class="dl-title" style="color:#a78bfa">&#x1F4CA; Abuse Detector</div>
        <div class="dl-row">Watches for <b>batch-size spikes &amp; drop-rate anomalies</b></div>
        <div class="dl-row">Fires when count ratio &gt; <b>""" + str(engine.abuse.spike_factor) + """&times;</b> baseline mean</div>
        <div class="dl-row">Also: drop-rate spike &amp; normalization-violation spike (Z &gt; 2.5)</div>
        <div class="dl-row">Stages: <b>all stages</b> (extra checks on preprocessing)</div>
      </div>
    </div>

    <div class="sec-lbl">Baseline Maturity &amp; Metric Comparison</div>
    """ + baseline_html + """

    <div class="sec-lbl">Stage Telemetry (latest reading)</div>
    <table>
      <thead><tr><th>Stage</th><th>Records</th><th>Key Metrics</th></tr></thead>
      <tbody>""" + stage_rows + """</tbody>
    </table>
  </div>

  <div>
    <div class="feed">
      <div class="feed-hdr">
        <span class="feed-title">&#x1F4E1; Live Activity</span>
        <span class="feed-ct">""" + str(feed_count) + """ events</span>
      </div>
      <div class="feed-body">""" + feed_html + """</div>
    </div>
  </div>

</div>

<div class="sb">
  <div class="sb-it">
    <span class="sb-dot """ + dot_cls + """"></span>
    <span class="sb-lbl">Status</span>
    <span class="sb-val">""" + sys_status + """</span>
  </div>
  <div class="sb-it"><span class="sb-lbl">Model</span><span class="sb-val">""" + OLLAMA_MODEL + """</span></div>
  <div class="sb-it"><span class="sb-lbl">Detection</span><span class="sb-val">""" + detect_mode + """</span></div>
  <div class="sb-it"><span class="sb-lbl">Sensitivity</span><span class="sb-val">""" + SENSITIVITY + """</span></div>
  <div class="sb-it"><span class="sb-lbl">Uptime</span><span class="sb-val">""" + uptime_str + """</span></div>
  <div class="sb-r">""" + render_time + """ UTC &middot; refreshes every 5s</div>
</div>

<div id="toast"></div>

<script>
""" + alert_detail_js + """
(function() {
  var cls = localStorage.getItem('pg-theme') || 'dark';
  document.documentElement.className = cls;
  var btn = document.getElementById('theme-btn');
  if (btn) btn.innerHTML = cls === 'dark' ? '&#x1F319;' : '&#x2600;&#xFE0F;';
})();

function toggleTheme() {
  var html = document.documentElement;
  var next = html.classList.contains('light') ? 'dark' : 'light';
  html.className = next;
  localStorage.setItem('pg-theme', next);
  document.getElementById('theme-btn').innerHTML = next === 'dark' ? '&#x1F319;' : '&#x2600;&#xFE0F;';
}

function showToast(msg, cls) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + cls;
  setTimeout(function() { t.className = ''; }, 3000);
}

async function triggerRun(mode) {
  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode})
    });
    if (!resp.ok) {
      const e = await resp.json().catch(function() { return {detail: 'Unknown error'}; });
      throw new Error(e.detail || 'Request failed');
    }
    const data = await resp.json();
    const n = (data.alerts_triggered || []).length;
    showToast(n > 0 ? n + ' alert(s) raised' : 'Clean run -- no threats', n > 0 ? 'err' : 'ok');
    setTimeout(function() { location.reload(); }, 900);
  } catch (err) {
    showToast('Error: ' + err.message, 'err');
  }
}

async function clearAlerts() {
  try {
    const resp = await fetch('/alerts', {method: 'DELETE'});
    if (!resp.ok) throw new Error('Failed to reset');
    showToast('Dashboard reset', 'ok');
    setTimeout(function() { location.reload(); }, 800);
  } catch (err) {
    showToast('Error: ' + err.message, 'err');
  }
}

function toggleDetail(id) {
  var row = document.getElementById('det-' + id);
  var wrap = document.getElementById('adw-' + id);
  var expandIcon = row.previousElementSibling.querySelector('.expand-icon');
  if (!row) return;
  var visible = row.style.display !== 'none';
  row.style.display = visible ? 'none' : '';
  if (expandIcon) expandIcon.parentElement.parentElement.classList.toggle('expanded', !visible);
  if (!visible && wrap && wrap.innerHTML === '') {
    var details = ALERT_DETAILS[id] || [];
    if (!details.length) { wrap.innerHTML = '<span style="color:var(--muted)">No sub-details available</span>'; return; }
    var html = '';
    details.forEach(function(d) {
      html += '<span class="ad-check">';
      Object.entries(d).forEach(function([k, v]) {
        var cls = k === 'z_score' || k === 'z_score' ? 'ad-z' : (k === 'metric' || k === 'check' ? 'ad-val' : 'ad-val');
        if (k === 'z_score') cls = 'ad-z';
        html += '<span class="ad-key">' + k + ':&nbsp;</span><span class="' + cls + '">' + v + '</span> &nbsp;';
      });
      html += '</span>';
    });
    wrap.innerHTML = html;
  }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard."""
    return _build_dashboard(
        app.state.engine,
        app.state.telemetry,
        app.state.runner,
        app.state.auto_run_enabled,
        app.state.warmup_done,
        app.state.start_time,
    )


@app.post("/run")
async def trigger_run(req: RunRequest):
    """Trigger a single pipeline run."""
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
    return app.state.engine.get_alerts(100)


@app.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    for alert in app.state.engine.alerts:
        if alert.id == alert_id:
            return alert.to_dict()
    raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found")


@app.get("/telemetry/{stage}")
async def get_telemetry(stage: str):
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
    return {
        "status": "ok",
        "ollama": app.state.ollama.is_available(),
        "model": OLLAMA_MODEL,
        "sensitivity": SENSITIVITY,
        "auto_run": app.state.auto_run_enabled,
        "warmup_done": app.state.warmup_done,
    }


@app.delete("/alerts")
async def clear_alerts():
    app.state.engine.clear_alerts()
    return {"cleared": True}


@app.get("/diagnostics")
async def diagnostics():
    """Return baseline sample counts, detector thresholds, and last run info."""
    engine: AnomalyEngine = app.state.engine
    snap = engine.baseline.snapshot()
    baseline_counts: dict = {}
    for stage, metrics in snap.items():
        baseline_counts[stage] = {m: len(v) for m, v in metrics.items()}

    return {
        "baseline_counts": baseline_counts,
        "detector_thresholds": {
            "poisoning_zscore": engine.poisoning.threshold,
            "adversarial_low_conf": engine.adversarial.low_conf_threshold,
            "adversarial_unknown_rate": engine.adversarial.unknown_rate_threshold,
            "adversarial_conf_std_mult": engine.adversarial.conf_std_multiplier,
            "abuse_spike_factor": engine.abuse.spike_factor,
        },
        "last_run": app.state.runner.last_run,
    }
