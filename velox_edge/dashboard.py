"""Single-page dashboard. FastAPI + vanilla JS + a tiny Chart.js dependency.

Endpoints:
  GET  /                     — the one HTML page
  GET  /api/status           — equity, today's P&L, lite mode flag, kill switch state
  GET  /api/equity-curve     — series of {ts, equity, spy_price} for charting
  GET  /api/scoreboard       — Claude vs GPT vs Consensus P&L attribution
  GET  /api/decisions        — last N consensus decisions
  GET  /api/skips            — last N skipped trades (where models disagreed / low conf)
  GET  /api/positions        — open positions from broker + ratchet state
  POST /api/kill             — flip TRADING_HALTED runtime flag
"""

from __future__ import annotations

import asyncio
import os
import time as _time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from velox_edge import broker, config, state


app = FastAPI(title="Velox Edge")

_kill_flag_runtime = bool(config.TRADING_HALTED)


def _check_token(request: Request):
    if not config.DASHBOARD_TOKEN:
        return  # no token = no auth (dev)
    token = request.query_params.get("token") or request.headers.get("x-dashboard-token", "")
    if token != config.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/api/status")
async def api_status(request: Request):
    _check_token(request)
    try:
        acct = await broker.get_account()
        equity = float(acct.get("equity") or 0)
        last_equity = float(acct.get("last_equity") or equity)
        day_pnl = equity - last_equity
        day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0
    except Exception:
        equity = 0
        day_pnl = 0
        day_pnl_pct = 0

    summary = state.trade_summary()
    return {
        "equity": equity,
        "day_pnl": day_pnl,
        "day_pnl_pct": day_pnl_pct,
        "trading_halted": _kill_flag_runtime or config.TRADING_HALTED,
        "summary": summary,
        "config": {
            "position_size_pct": config.POSITION_SIZE_PCT,
            "max_positions": config.MAX_CONCURRENT_POSITIONS,
            "min_consensus_confidence": config.MIN_CONSENSUS_CONFIDENCE,
            "ratchet": {
                "hard_stop_pct": config.RATCHET_HARD_STOP_PCT,
                "activation_pct": config.RATCHET_ACTIVATION_PCT,
                "trail_pct": config.RATCHET_TRAIL_PCT,
                "min_hold_seconds": config.RATCHET_MIN_HOLD_SECONDS,
            },
            "anthropic_model": config.ANTHROPIC_MODEL,
            "openai_model": config.OPENAI_MODEL,
        },
    }


@app.get("/api/equity-curve")
async def api_equity_curve(request: Request):
    _check_token(request)
    return state.equity_history()


@app.get("/api/scoreboard")
async def api_scoreboard(request: Request):
    _check_token(request)
    return state.model_scoreboard()


@app.get("/api/decisions")
async def api_decisions(request: Request, limit: int = 30):
    _check_token(request)
    return state.recent_decisions(limit=min(200, max(1, limit)))


@app.get("/api/skips")
async def api_skips(request: Request, limit: int = 30):
    _check_token(request)
    return state.recent_skips(limit=min(500, max(1, limit)))


@app.get("/api/trades")
async def api_trades(request: Request, limit: int = 50):
    _check_token(request)
    return state.recent_closed_trades(limit=min(200, max(1, limit)))


@app.get("/api/positions")
async def api_positions(request: Request):
    _check_token(request)
    try:
        positions = await broker.get_positions()
    except Exception as e:
        return {"error": str(e), "positions": []}
    out = []
    for p in positions:
        qty = float(p.get("qty") or 0)
        avg = float(p.get("avg_entry_price") or 0)
        current = float(p.get("current_price") or 0)
        unrealized = float(p.get("unrealized_pl") or 0)
        unrealized_pct = float(p.get("unrealized_plpc") or 0) * 100
        out.append({
            "symbol": p.get("symbol"),
            "side": "long" if qty > 0 else "short",
            "qty": qty,
            "entry": avg,
            "current": current,
            "unrealized": unrealized,
            "unrealized_pct": unrealized_pct,
        })
    return {"positions": out}


@app.post("/api/kill")
async def api_kill(request: Request):
    _check_token(request)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    halted = bool(body.get("halted", True))
    global _kill_flag_runtime
    _kill_flag_runtime = halted
    config.TRADING_HALTED = halted  # propagate to in-process consumers
    state.audit("kill_switch", "warn", f"trading_halted={halted}")
    return {"trading_halted": halted}


@app.get("/api/audit")
async def api_audit(request: Request, limit: int = 50):
    _check_token(request)
    return state.recent_audit(limit=min(500, max(1, limit)))


@app.get("/api/market-brief")
async def api_market_brief(request: Request):
    _check_token(request)
    brief = state.latest_market_brief()
    if not brief:
        return {"available": False}
    return {
        "available": True,
        "timestamp": brief["timestamp"],
        "session_label": brief.get("session_label", ""),
        "text": brief["text"],
        "citations": brief.get("citations", []),
        "error": brief.get("error", ""),
    }


@app.get("/api/daily-review")
async def api_daily_review(request: Request):
    _check_token(request)
    r = state.latest_daily_review()
    if not r:
        return {"available": False}
    return {
        "available": True,
        "timestamp": r["timestamp"],
        "review_date": r.get("review_date", ""),
        "text": r["text"],
        "day_pnl": r.get("day_pnl", 0),
        "n_closed": r.get("n_closed", 0),
    }


# ── The single HTML page ───────────────────────────────────────────


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Velox Edge</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap">
<style>
  /* ─── Velox Edge · v1 ──────────────────────────────────────────────
     Design language: editorial, restrained, warm-paper-on-deep-warmth.
     Voice: a quarterly letter from a steward of capital, not a casino.
     Palette: ivory ink on warm graphite. One gold accent. Muted signal.
  ─────────────────────────────────────────────────────────────────── */

  :root {
    --bg:        #0a0a0b;          /* deep warm graphite, not GitHub gray */
    --panel:    #111114;
    --panel-2:  #16161a;
    --hairline: rgba(232, 226, 213, 0.08);
    --hairline-strong: rgba(232, 226, 213, 0.16);
    --ink:      #ece6d8;           /* warm ivory primary text */
    --ink-soft: #aaa39a;
    --ink-mute: #6e6a62;
    --gold:     #d97a4a;          /* Edge: copper / hunter's brass — sharper than vanilla gold */
    --gold-soft: rgba(217, 122, 74, 0.15);
    --win:      #7fb491;           /* muted forest */
    --win-soft: rgba(127, 180, 145, 0.12);
    --loss:     #c47866;           /* warm earth red */
    --loss-soft: rgba(196, 120, 102, 0.12);
    --warn:     #d4b07a;
    --serif: 'Fraunces', 'Hoefler Text', 'Iowan Old Style', 'Apple Garamond', 'Baskerville', Georgia, serif;
    --sans:  'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono:  'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    --tracking-loose: 0.18em;
    --tracking-mid: 0.08em;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background:
      radial-gradient(ellipse 1200px 600px at 50% -300px, rgba(201,168,112,0.06), transparent 70%),
      var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    font-weight: 400;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    line-height: 1.55;
    min-height: 100vh;
  }
  ::selection { background: var(--gold-soft); color: var(--ink); }

  .container {
    max-width: 1180px;
    margin: 0 auto;
    padding: 56px 40px 80px;
  }

  /* ─── Top bar ─────────────────────────────────────────────────── */
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--sans);
    font-size: 11px;
    color: var(--ink-mute);
    letter-spacing: var(--tracking-mid);
    text-transform: uppercase;
    border-bottom: 1px solid var(--hairline);
    padding-bottom: 18px;
    margin-bottom: 64px;
  }
  .topbar .left { display: flex; gap: 28px; }
  .topbar .left .session-now { color: var(--gold); }
  .topbar .right .clock { color: var(--ink-soft); font-variant-numeric: tabular-nums; }

  /* ─── Hero ────────────────────────────────────────────────────── */
  .hero {
    text-align: center;
    margin-bottom: 88px;
  }
  .monogram {
    font-family: var(--serif);
    font-weight: 300;
    font-size: 56px;
    color: var(--gold);
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 10px;
    font-feature-settings: 'ss01' on;
  }
  .wordmark {
    font-family: var(--serif);
    font-weight: 400;
    font-size: 13px;
    color: var(--ink-soft);
    letter-spacing: var(--tracking-loose);
    text-transform: uppercase;
    margin-bottom: 60px;
  }
  .hero-equity {
    font-family: var(--serif);
    font-weight: 300;
    font-size: clamp(72px, 11vw, 132px);
    line-height: 0.95;
    color: var(--ink);
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
    margin-bottom: 18px;
  }
  .hero-equity .currency { color: var(--ink-mute); font-size: 0.55em; vertical-align: 0.45em; padding-right: 4px; }
  .hero-equity .cents { color: var(--ink-mute); font-weight: 300; }
  .hero-meta {
    display: flex;
    justify-content: center;
    gap: 60px;
    font-family: var(--sans);
    font-size: 13px;
    color: var(--ink-soft);
    margin-top: 14px;
  }
  .hero-meta .v {
    font-family: var(--mono);
    font-size: 17px;
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.01em;
    display: block;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
  }
  .hero-meta .l {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: var(--tracking-loose);
    color: var(--ink-mute);
  }
  .hero-meta .v.win  { color: var(--win); }
  .hero-meta .v.loss { color: var(--loss); }

  /* ─── The mission line ───────────────────────────────────────── */
  .mission {
    text-align: center;
    margin: 96px auto 96px;
    max-width: 720px;
    border-top: 1px solid var(--hairline);
    border-bottom: 1px solid var(--hairline);
    padding: 36px 24px;
  }
  .mission .quote {
    font-family: var(--serif);
    font-weight: 300;
    font-style: italic;
    font-size: 21px;
    line-height: 1.5;
    color: var(--ink-soft);
    letter-spacing: -0.005em;
  }
  .mission .quote::before { content: '“'; color: var(--gold); padding-right: 4px; }
  .mission .quote::after  { content: '”'; color: var(--gold); padding-left: 2px; }
  .mission .attrib {
    margin-top: 18px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: var(--tracking-loose);
    color: var(--ink-mute);
  }

  /* ─── Section header ────────────────────────────────────────── */
  .section-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 28px;
  }
  .section-head .dot { width: 6px; height: 6px; background: var(--gold); border-radius: 50%; }
  .section-head h2 {
    font-family: var(--serif);
    font-weight: 400;
    font-size: 22px;
    color: var(--ink);
    margin: 0;
    letter-spacing: -0.01em;
  }
  .section-head .sub {
    font-size: 11px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
    margin-left: auto;
    font-family: var(--sans);
  }
  .section { margin-bottom: 88px; }

  /* ─── Daily review (the editorial voice) ───────────────────── */
  .review-section {
    margin: 0 auto 88px;
    max-width: 760px;
  }
  .review-eyebrow {
    text-align: center;
    font-family: var(--sans);
    font-size: 11px;
    color: var(--gold);
    text-transform: uppercase;
    letter-spacing: var(--tracking-loose);
    margin-bottom: 18px;
  }
  .review-headline {
    text-align: center;
    font-family: var(--serif);
    font-weight: 400;
    font-size: 28px;
    color: var(--ink);
    letter-spacing: -0.015em;
    margin-bottom: 32px;
    line-height: 1.2;
  }
  .review-body {
    font-family: var(--serif);
    font-weight: 300;
    font-size: 17px;
    line-height: 1.8;
    color: var(--ink-soft);
    letter-spacing: -0.005em;
  }
  .review-body p { margin: 0 0 18px; }
  .review-meta {
    margin-top: 24px;
    text-align: center;
    font-family: var(--sans);
    font-size: 11px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
  }

  /* ─── Market brief ──────────────────────────────────────────── */
  .brief-card {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-left: 2px solid var(--gold);
    border-radius: 2px;
    padding: 28px 32px;
  }
  .brief-text {
    font-family: var(--serif);
    font-weight: 300;
    font-size: 15.5px;
    line-height: 1.7;
    color: var(--ink);
    letter-spacing: -0.005em;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .brief-text strong {
    color: var(--gold);
    font-weight: 500;
  }
  .brief-text em { color: var(--ink-soft); font-style: italic; }
  .brief-text ul, .brief-text ol { padding-left: 22px; margin: 8px 0; }
  .brief-text li { margin-bottom: 6px; }
  .brief-text p { margin: 10px 0; }
  .brief-text h1, .brief-text h2, .brief-text h3, .brief-text h4 {
    font-family: var(--serif);
    font-weight: 500;
    font-size: 13px;
    color: var(--gold);
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
    margin: 18px 0 8px;
  }
  .brief-citations {
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid var(--hairline);
    font-family: var(--sans);
    font-size: 11px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
  }
  .brief-citations a {
    color: var(--ink-soft);
    text-decoration: none;
    margin-right: 18px;
    border-bottom: 1px solid var(--hairline-strong);
    transition: color 200ms ease;
  }
  .brief-citations a:hover { color: var(--gold); }

  /* ─── The chart ─────────────────────────────────────────────── */
  .chart-wrap {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 2px;
    padding: 32px 28px 24px;
  }
  .chart-legend {
    display: flex;
    gap: 32px;
    margin-bottom: 18px;
    font-family: var(--sans);
    font-size: 11px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
  }
  .chart-legend .swatch { display: inline-block; width: 14px; height: 2px; margin-right: 8px; vertical-align: middle; }
  .chart-legend .velox-swatch { background: var(--gold); }
  .chart-legend .spy-swatch { background: var(--ink-soft); border-top: 1px dashed var(--ink-soft); height: 0; }

  /* ─── Duel cards ────────────────────────────────────────────── */
  .duel {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--hairline);
    border: 1px solid var(--hairline);
    border-radius: 2px;
    overflow: hidden;
  }
  @media (max-width: 800px) { .duel { grid-template-columns: 1fr; } }
  .duel-cell {
    background: var(--panel);
    padding: 28px 26px;
    position: relative;
  }
  .duel-cell.consensus { background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); }
  .duel-cell .label {
    font-family: var(--sans);
    font-size: 10px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: var(--tracking-loose);
    margin-bottom: 22px;
  }
  .duel-cell .name {
    font-family: var(--serif);
    font-size: 26px;
    font-weight: 400;
    color: var(--ink);
    margin-bottom: 18px;
    letter-spacing: -0.01em;
  }
  .duel-cell.consensus .name { color: var(--gold); }
  .duel-cell .pnl {
    font-family: var(--mono);
    font-size: 32px;
    font-weight: 300;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }
  .duel-cell .pnl.win { color: var(--win); }
  .duel-cell .pnl.loss { color: var(--loss); }
  .duel-cell .meta {
    font-family: var(--sans);
    font-size: 11px;
    color: var(--ink-mute);
    letter-spacing: 0.02em;
  }
  .duel-cell .meta strong { color: var(--ink-soft); font-weight: 500; }

  /* ─── Editorial entries (decisions / skips / trades) ─────────── */
  .entry-list { border-top: 1px solid var(--hairline); }
  .entry {
    display: grid;
    grid-template-columns: 80px 90px 1fr 110px;
    gap: 20px;
    padding: 18px 0;
    border-bottom: 1px solid var(--hairline);
    align-items: baseline;
    font-size: 13px;
  }
  .entry .time {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-mute);
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .entry .symbol {
    font-family: var(--serif);
    font-size: 17px;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.005em;
  }
  .entry .body {
    color: var(--ink-soft);
    font-size: 13px;
    line-height: 1.5;
  }
  .entry .body .vote {
    color: var(--ink);
    font-family: var(--mono);
    font-size: 11px;
    margin-right: 14px;
  }
  .entry .body .vote-claude { color: var(--gold); }
  .entry .body .vote-gpt { color: #87b3c9; }
  .entry .body .reason { color: var(--ink-mute); font-style: italic; display: block; margin-top: 4px; font-size: 12px; }
  .entry .verdict {
    font-family: var(--sans);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: var(--tracking-mid);
    text-align: right;
  }
  .entry .verdict.executed { color: var(--win); }
  .entry .verdict.skipped { color: var(--ink-mute); }
  .entry .verdict.win { color: var(--win); }
  .entry .verdict.loss { color: var(--loss); }
  .entry .verdict .pnl-line {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 500;
    margin-top: 2px;
    display: block;
    font-variant-numeric: tabular-nums;
    text-transform: none;
    letter-spacing: 0;
  }

  /* ─── Positions table (still tabular) ─────────────────────── */
  .positions-table { width: 100%; border-collapse: collapse; }
  .positions-table th, .positions-table td {
    padding: 16px 12px; text-align: left;
    border-bottom: 1px solid var(--hairline);
    font-family: var(--mono);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  .positions-table th {
    color: var(--ink-mute);
    font-family: var(--sans);
    font-weight: 500;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: var(--tracking-loose);
    border-bottom: 1px solid var(--hairline-strong);
  }
  .positions-table td.symbol {
    font-family: var(--serif);
    font-size: 16px;
    font-weight: 500;
    color: var(--ink);
  }
  .positions-table .win { color: var(--win); }
  .positions-table .loss { color: var(--loss); }
  .positions-table .empty {
    text-align: center; color: var(--ink-mute);
    font-family: var(--serif); font-style: italic;
    padding: 36px 0;
  }

  /* ─── Halt control ─────────────────────────────────────────── */
  .halt-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-top: 48px;
    padding-top: 28px;
    border-top: 1px solid var(--hairline);
  }
  .halt-btn {
    background: transparent;
    color: var(--ink-soft);
    border: 1px solid var(--hairline-strong);
    padding: 11px 22px;
    border-radius: 1px;
    font-family: var(--sans);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: var(--tracking-loose);
    text-transform: uppercase;
    cursor: pointer;
    transition: all 200ms ease;
  }
  .halt-btn:hover { border-color: var(--loss); color: var(--loss); }
  .halt-btn.halted { color: var(--win); border-color: rgba(127,180,145,0.4); }
  .halt-btn.halted:hover { background: var(--win-soft); }
  .halt-note { font-size: 12px; color: var(--ink-mute); font-family: var(--sans); }

  .halt-banner {
    display: none;
    margin-top: 24px;
    padding: 16px 22px;
    border-left: 2px solid var(--loss);
    background: var(--loss-soft);
    color: var(--ink);
    font-family: var(--serif);
    font-size: 14px;
    font-style: italic;
  }

  /* ─── Sessions strip ───────────────────────────────────────── */
  .sessions-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 28px;
  }
  @media (max-width: 700px) { .sessions-strip { grid-template-columns: repeat(2, 1fr); } }
  .session-pill {
    background: var(--panel);
    border: 1px solid var(--hairline);
    padding: 14px 16px;
    border-radius: 2px;
    font-family: var(--sans);
  }
  .session-pill.upcoming { border-color: var(--gold-soft); }
  .session-pill.upcoming .name { color: var(--gold); }
  .session-pill.completed { opacity: 0.6; }
  .session-pill .when {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-mute);
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .session-pill .name {
    font-family: var(--serif);
    font-size: 14px;
    color: var(--ink);
    text-transform: capitalize;
    letter-spacing: -0.005em;
  }

  /* ─── Footer ──────────────────────────────────────────────── */
  footer {
    margin-top: 120px;
    padding-top: 32px;
    border-top: 1px solid var(--hairline);
    text-align: center;
    font-size: 11px;
    color: var(--ink-mute);
    font-family: var(--sans);
    letter-spacing: 0.04em;
    line-height: 2;
  }
  footer .colophon {
    font-family: var(--serif);
    font-style: italic;
    font-size: 13px;
    color: var(--ink-soft);
    letter-spacing: -0.005em;
    margin-bottom: 14px;
    font-weight: 300;
  }

  .skel { color: var(--ink-mute); font-style: italic; font-family: var(--serif); }
</style>
</head>
<body>
<div class="container">

  <!-- TOP BAR -->
  <div class="topbar">
    <div class="left">
      <span id="topStatus">Loading…</span>
      <span id="topNextSession" class="session-now"></span>
    </div>
    <div class="right">
      <span class="clock" id="topClock">—</span>
    </div>
  </div>

  <!-- HERO -->
  <div class="hero">
    <div class="monogram">V</div>
    <div class="wordmark">Velox &nbsp;·&nbsp; Core</div>
    <div class="hero-equity" id="heroEquity"><span class="currency">$</span>—<span class="cents">.—</span></div>
    <div class="hero-meta">
      <div>
        <span class="v" id="heroDayPnl">—</span>
        <span class="l">Today</span>
      </div>
      <div>
        <span class="v" id="heroVsSpy">—</span>
        <span class="l">vs SPY</span>
      </div>
      <div>
        <span class="v" id="heroTrades">—</span>
        <span class="l">Closed Trades</span>
      </div>
      <div>
        <span class="v" id="heroWinRate">—</span>
        <span class="l">Win Rate</span>
      </div>
    </div>
  </div>

  <!-- MISSION -->
  <div class="mission">
    <div class="quote">When the crowd runs in, walk out. When the crowd flees, walk in. The market pays you for taking the trade no one else will.</div>
    <div class="attrib">Velox Edge · The Contrarian Mandate</div>
  </div>

  <!-- DAILY REVIEW (the editorial voice — appears once written) -->
  <div class="review-section" id="reviewSection" style="display:none">
    <div class="review-eyebrow">Today, in 200 words</div>
    <div class="review-headline" id="reviewHeadline">—</div>
    <div class="review-body" id="reviewBody"></div>
    <div class="review-meta" id="reviewMeta"></div>
  </div>

  <!-- SESSIONS -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>Today's sessions</h2>
      <span class="sub" id="sessionsCount">—</span>
    </div>
    <div class="sessions-strip" id="sessionsStrip"></div>
  </div>

  <!-- MARKET BRIEF -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>The market brief</h2>
      <span class="sub" id="briefMeta">awaiting first session</span>
    </div>
    <div class="brief-card" id="briefCard">
      <div class="brief-text skel">Perplexity will publish a fresh brief at the next session — what the tape looks like right now, which of our 40 tickers have material catalysts, what regime tone we're in. Both Claude and GPT see this brief before they vote.</div>
      <div class="brief-citations" id="briefCitations"></div>
    </div>
  </div>

  <!-- EQUITY VS SPY -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>Equity, vs the index</h2>
      <span class="sub" id="alphaLabel">—</span>
    </div>
    <div class="chart-wrap">
      <div class="chart-legend">
        <span><span class="swatch velox-swatch"></span>Velox Edge</span>
        <span><span class="swatch spy-swatch"></span>SPY · normalized</span>
      </div>
      <canvas id="equityChart" height="240"></canvas>
    </div>
  </div>

  <!-- THE DUEL -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>The duel</h2>
      <span class="sub">When each model was on the right side of a trade</span>
    </div>
    <div class="duel" id="duelGrid">
      <div class="duel-cell"><div class="label">Claude</div><div class="name">Opus 4.7</div><div class="pnl">—</div><div class="meta skel">awaiting first session</div></div>
      <div class="duel-cell"><div class="label">OpenAI</div><div class="name">GPT 5.4</div><div class="pnl">—</div><div class="meta skel">awaiting first session</div></div>
      <div class="duel-cell consensus"><div class="label">Both agreed</div><div class="name">Consensus</div><div class="pnl">—</div><div class="meta skel">where the real edge lives</div></div>
    </div>
  </div>

  <!-- POSITIONS -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>Open positions</h2>
      <span class="sub" id="positionsCount">—</span>
    </div>
    <table class="positions-table"><thead>
      <tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>P&amp;L</th><th>%</th></tr>
    </thead><tbody id="positionsBody"></tbody></table>
  </div>

  <!-- DECISIONS -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>The journal</h2>
      <span class="sub">Every decision, recorded</span>
    </div>
    <div class="entry-list" id="decisionsList"></div>
  </div>

  <!-- SKIP LOG -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>The skip log</h2>
      <span class="sub">What we did not do, and why</span>
    </div>
    <div class="entry-list" id="skipsList"></div>
  </div>

  <!-- CLOSED TRADES -->
  <div class="section">
    <div class="section-head">
      <span class="dot"></span>
      <h2>Closed trades</h2>
      <span class="sub">Attribution: who was right</span>
    </div>
    <div class="entry-list" id="tradesList"></div>
  </div>

  <!-- HALT CONTROL -->
  <div class="section">
    <div class="halt-row">
      <button class="halt-btn" id="haltBtn" onclick="toggleKill()">Halt trading</button>
      <span class="halt-note">Existing positions remain protected by the ratchet. New entries blocked until resumed.</span>
    </div>
    <div class="halt-banner" id="haltBanner">Trading is halted. The ratchet is still managing every open position. New entries are blocked until you resume.</div>
  </div>

  <!-- FOOTER -->
  <footer>
    <div class="colophon">For Holly, Evelyn, Emma, and Miles.</div>
    <div id="footerConfig">—</div>
    <div>Paper account · Anthropic + OpenAI consensus · No paid feeds · Single desk</div>
  </footer>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const Q = TOKEN ? '?token=' + encodeURIComponent(TOKEN) : '';

async function api(path) {
  const r = await fetch(path + Q);
  if (!r.ok) throw new Error(path + ' ' + r.status);
  return r.json();
}

// ── Formatters ─────────────────────────────────────────────────
function fmtMoney(n, sign=true) {
  const v = Math.abs(n);
  const s = sign ? (n>=0 ? '+' : '−') : '';
  return s + '$' + v.toFixed(2);
}
function fmtPct(n) { return (n>=0?'+':'−') + Math.abs(n).toFixed(2) + '%'; }
function fmtTime(ts) {
  const d = new Date(ts*1000);
  return d.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'});
}
function relTime(ts) {
  const s = Math.floor(Date.now()/1000 - ts);
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

// ── Top bar clock + status ─────────────────────────────────────
const SESSIONS = [
  {h: 9, m: 35, name: 'open'},
  {h: 11, m: 0, name: 'mid morning'},
  {h: 13, m: 30, name: 'lunch reversal'},
  {h: 15, m: 0, name: 'power hour'},
  {h: 15, m: 45, name: 'pre-close'},
];
const EOD_FLATTEN = {h: 15, m: 55};

function nowEt() {
  const d = new Date();
  // Convert to ET via toLocaleString round-trip
  const etStr = d.toLocaleString('en-US', {timeZone: 'America/New_York'});
  return new Date(etStr);
}

function nextSessionInfo() {
  const now = nowEt();
  const today = SESSIONS.map(s => {
    const t = new Date(now);
    t.setHours(s.h, s.m, 0, 0);
    return {time: t, name: s.name};
  });
  const upcoming = today.find(s => s.time > now);
  if (upcoming) {
    const diff = upcoming.time - now;
    const hours = Math.floor(diff / 3600000);
    const mins = Math.floor((diff % 3600000) / 60000);
    const wait = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
    return {label: `Next: ${upcoming.name} in ${wait}`, today, idx: today.indexOf(upcoming)};
  }
  return {label: 'Sessions complete · next: tomorrow 9:35 AM ET', today, idx: -1};
}

function updateTopBar() {
  const et = nowEt();
  const clockStr = et.toLocaleTimeString('en-US', {hour:'numeric', minute:'2-digit', hour12:true}) + ' ET';
  document.getElementById('topClock').textContent = clockStr;
  const info = nextSessionInfo();
  document.getElementById('topNextSession').textContent = info.label;
}
setInterval(updateTopBar, 30000);
updateTopBar();

// ── Sessions strip ─────────────────────────────────────────────
function renderSessions() {
  const info = nextSessionInfo();
  const html = info.today.map((s, i) => {
    const past = s.time < nowEt();
    const isNext = i === info.idx;
    const cls = past ? 'completed' : (isNext ? 'upcoming' : '');
    const time = s.time.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'});
    return `<div class="session-pill ${cls}">
      <div class="when">${time}</div>
      <div class="name">${s.name}</div>
    </div>`;
  }).join('');
  document.getElementById('sessionsStrip').innerHTML = html;
  const completed = info.today.filter(s => s.time < nowEt()).length;
  document.getElementById('sessionsCount').textContent = `${completed} of 5 complete`;
}

// ── Status (hero) ──────────────────────────────────────────────
async function refreshStatus() {
  const s = await api('/api/status');
  // Hero equity
  const eq = (s.equity||0);
  const whole = Math.floor(eq).toLocaleString();
  const cents = '.' + (eq - Math.floor(eq)).toFixed(2).slice(2);
  document.getElementById('heroEquity').innerHTML =
    `<span class="currency">$</span>${whole}<span class="cents">${cents}</span>`;

  const dayCls = s.day_pnl > 0 ? 'win' : (s.day_pnl < 0 ? 'loss' : '');
  document.getElementById('heroDayPnl').textContent = fmtMoney(s.day_pnl) + '  ·  ' + fmtPct(s.day_pnl_pct);
  document.getElementById('heroDayPnl').className = 'v ' + dayCls;

  // vsSpy is computed from equity-curve refresh
  document.getElementById('heroTrades').textContent = (s.summary?.total_trades || 0);
  document.getElementById('heroWinRate').textContent = (s.summary?.win_rate || 0).toFixed(0) + '%';

  document.getElementById('topStatus').textContent =
    (s.trading_halted ? 'TRADING HALTED' : 'TRADING LIVE') +
    `  ·  Paper · $${(s.equity||0).toFixed(0)}`;

  document.getElementById('footerConfig').textContent =
    `${s.config.anthropic_model} + ${s.config.openai_model}  ·  consensus ≥ ${s.config.min_consensus_confidence}%  ·  ${s.config.position_size_pct}% per position, max ${s.config.max_positions} concurrent  ·  ratchet ${s.config.ratchet.hard_stop_pct}% / +${s.config.ratchet.activation_pct}% / ${s.config.ratchet.trail_pct}% trail`;

  const banner = document.getElementById('haltBanner');
  const btn = document.getElementById('haltBtn');
  if (s.trading_halted) {
    banner.style.display = 'block';
    btn.textContent = 'Resume trading';
    btn.classList.add('halted');
  } else {
    banner.style.display = 'none';
    btn.textContent = 'Halt trading';
    btn.classList.remove('halted');
  }
}

// ── Equity chart + alpha ───────────────────────────────────────
let equityChart;
async function refreshEquity() {
  const data = await api('/api/equity-curve');
  if (!data.length) {
    document.getElementById('alphaLabel').textContent = 'awaiting data';
    return;
  }
  const labels = data.map(d => new Date(d.timestamp*1000).toLocaleDateString('en-US', {month:'short', day:'numeric'}));
  const equityVals = data.map(d => d.equity);
  const firstEq = equityVals[0] || 1;
  const firstSpy = data.find(d => d.spy)?.spy || 0;
  const spyVals = firstSpy ? data.map(d => d.spy ? (d.spy / firstSpy * firstEq) : null) : [];

  // Compute alpha
  const lastEq = equityVals[equityVals.length - 1];
  const eqRet = ((lastEq / firstEq) - 1) * 100;
  let label = `Velox ${fmtPct(eqRet)}`;
  if (firstSpy && spyVals.length) {
    const lastSpy = spyVals.filter(v => v != null).slice(-1)[0];
    const spyRet = lastSpy ? ((lastSpy / firstEq) - 1) * 100 : 0;
    const alpha = eqRet - spyRet;
    label = `Velox ${fmtPct(eqRet)}  ·  SPY ${fmtPct(spyRet)}  ·  Alpha ${fmtPct(alpha)}`;
    document.getElementById('heroVsSpy').textContent = fmtPct(alpha);
    document.getElementById('heroVsSpy').className = 'v ' + (alpha >= 0 ? 'win' : 'loss');
  }
  document.getElementById('alphaLabel').textContent = label;

  const ctx = document.getElementById('equityChart').getContext('2d');
  const cfg = {
    type: 'line',
    data: { labels, datasets: [
      { label: 'Velox', data: equityVals, borderColor: '#c9a870', backgroundColor: 'rgba(201,168,112,0.05)', tension: .25, pointRadius: 0, borderWidth: 1.5, fill: true },
      { label: 'SPY', data: spyVals, borderColor: '#aaa39a', borderDash: [3,5], tension: .25, pointRadius: 0, borderWidth: 1 },
    ]},
    options: {
      animation: false, responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#6e6a62', font: {size: 10, family: 'JetBrains Mono'}, maxTicksLimit: 8 }, grid: { display: false }, border: { color: 'rgba(232,226,213,0.16)' } },
        y: { ticks: { color: '#6e6a62', font: {size: 10, family: 'JetBrains Mono'}, callback: v => '$' + v.toLocaleString() }, grid: { color: 'rgba(232,226,213,0.04)' }, border: { display: false } },
      },
    },
  };
  if (equityChart) { equityChart.data = cfg.data; equityChart.update('none'); }
  else equityChart = new Chart(ctx, cfg);
}

// ── Duel (scoreboard) ──────────────────────────────────────────
async function refreshScoreboard() {
  const sb = await api('/api/scoreboard');
  const cell = (label, displayName, s) => {
    const cls = s.total_pnl > 0 ? 'win' : (s.total_pnl < 0 ? 'loss' : '');
    if (!s.trades) {
      return `<div class="duel-cell ${label==='Both agreed'?'consensus':''}">
        <div class="label">${label}</div>
        <div class="name">${displayName}</div>
        <div class="pnl">—</div>
        <div class="meta skel">awaiting first decision</div>
      </div>`;
    }
    return `<div class="duel-cell ${label==='Both agreed'?'consensus':''}">
      <div class="label">${label}</div>
      <div class="name">${displayName}</div>
      <div class="pnl ${cls}">${fmtMoney(s.total_pnl)}</div>
      <div class="meta"><strong>${s.trades}</strong> trades · <strong>${s.wins}</strong> wins · ${s.win_rate.toFixed(0)}% rate</div>
    </div>`;
  };
  // Pull display names from the configured models so the duel cards stay honest.
  const status = await api('/api/status');
  const claudeName = (status.config.anthropic_model||'claude').replace('claude-','').replace(/-/g,' ');
  const gptName = (status.config.openai_model||'gpt');
  document.getElementById('duelGrid').innerHTML =
    cell('Claude', claudeName, sb.claude) +
    cell('OpenAI', gptName, sb.gpt) +
    cell('Both agreed', 'Consensus', sb.consensus);
}

// ── Positions ──────────────────────────────────────────────────
async function refreshPositions() {
  const data = await api('/api/positions');
  const tbody = document.getElementById('positionsBody');
  document.getElementById('positionsCount').textContent =
    data.positions.length ? `${data.positions.length} open` : 'none open';
  if (!data.positions.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No positions yet — patient capital, not idle capital.</td></tr>';
    return;
  }
  tbody.innerHTML = data.positions.map(p => {
    const cls = p.unrealized >= 0 ? 'win' : 'loss';
    return `<tr>
      <td class="symbol">${p.symbol}</td>
      <td>${p.side === 'long' ? 'long' : 'short'}</td>
      <td>${p.qty}</td>
      <td>$${p.entry.toFixed(2)}</td>
      <td>$${p.current.toFixed(2)}</td>
      <td class="${cls}">${fmtMoney(p.unrealized)}</td>
      <td class="${cls}">${fmtPct(p.unrealized_pct)}</td>
    </tr>`;
  }).join('');
}

// ── Editorial entry renderer ───────────────────────────────────
function entryHtml(d, opts={}) {
  const verdictHtml = opts.verdictHtml || '';
  const claudeR = d.claude_reason ? d.claude_reason : '';
  const gptR = d.gpt_reason ? d.gpt_reason : '';
  const reasons = [];
  if (claudeR) reasons.push(`<em>Claude:</em> ${claudeR}`);
  if (gptR) reasons.push(`<em>GPT:</em> ${gptR}`);
  return `<div class="entry">
    <div class="time">${fmtTime(d.timestamp)}</div>
    <div class="symbol">${d.symbol}</div>
    <div class="body">
      <span class="vote vote-claude">C ${d.claude_action} · ${(d.claude_confidence||0).toFixed(0)}</span>
      <span class="vote vote-gpt">G ${d.gpt_action} · ${(d.gpt_confidence||0).toFixed(0)}</span>
      ${reasons.length ? `<span class="reason">${reasons.join('  ·  ')}</span>` : ''}
    </div>
    <div class="verdict ${opts.verdictClass||''}">${verdictHtml}</div>
  </div>`;
}

async function refreshDecisions() {
  const data = await api('/api/decisions?limit=20');
  const list = document.getElementById('decisionsList');
  if (!data.length) {
    list.innerHTML = '<div class="entry"><div class="time">—</div><div></div><div class="body skel">No decisions yet. The first session will fill this column with everything Claude and GPT thought, agreed on, and disagreed about.</div><div></div></div>';
    return;
  }
  list.innerHTML = data.map(d => {
    const verdictClass = d.executed ? 'executed' : 'skipped';
    const verdictHtml = d.executed
      ? `${d.consensus_action} · ${(d.consensus_confidence||0).toFixed(0)}%<span class="pnl-line">entered</span>`
      : `skipped<span class="pnl-line" style="color:var(--ink-mute);font-size:11px">${d.skip_reason||'no signal'}</span>`;
    return entryHtml(d, {verdictClass, verdictHtml});
  }).join('');
}

async function refreshSkips() {
  const data = await api('/api/skips?limit=20');
  const list = document.getElementById('skipsList');
  if (!data.length) {
    list.innerHTML = '<div class="entry"><div class="time">—</div><div></div><div class="body skel">No skips yet. When Claude and GPT disagree, those moments land here. Over thirty days this becomes the data ARC never measured.</div><div></div></div>';
    return;
  }
  list.innerHTML = data.map(d => entryHtml(d, {
    verdictClass: 'skipped',
    verdictHtml: `<span style="color:var(--warn)">skipped</span><span class="pnl-line" style="color:var(--ink-mute);font-size:11px">${d.skip_reason||'no consensus'}</span>`,
  })).join('');
}

async function refreshTrades() {
  const data = await api('/api/trades?limit=20');
  const list = document.getElementById('tradesList');
  if (!data.length) {
    list.innerHTML = '<div class="entry"><div class="time">—</div><div></div><div class="body skel">No closed trades yet. When the ratchet completes its first cycle, the verdict on each trade — and which model was on the right side — appears here.</div><div></div></div>';
    return;
  }
  list.innerHTML = data.map(t => {
    const cls = (t.pnl||0) >= 0 ? 'win' : 'loss';
    const hold = t.hold_seconds ? Math.round(t.hold_seconds/60) + 'm' : '—';
    return `<div class="entry">
      <div class="time">${fmtTime(t.exit_time||t.entry_time)}</div>
      <div class="symbol">${t.symbol}</div>
      <div class="body">
        <span class="vote vote-claude">C ${t.claude_vote||'?'}</span>
        <span class="vote vote-gpt">G ${t.gpt_vote||'?'}</span>
        <span class="reason">${t.side} · entry $${(t.entry_price||0).toFixed(2)} → exit $${(t.exit_price||0).toFixed(2)} · held ${hold} · ${t.exit_reason||''}</span>
      </div>
      <div class="verdict ${cls}">
        ${fmtPct(t.pnl_pct||0)}
        <span class="pnl-line">${fmtMoney(t.pnl||0)}</span>
      </div>
    </div>`;
  }).join('');
}

// ── Halt switch ────────────────────────────────────────────────
async function toggleKill() {
  const s = await api('/api/status');
  const r = await fetch('/api/kill' + Q, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({halted: !s.trading_halted}),
  });
  if (r.ok) refreshAll();
}

// ── Market brief ───────────────────────────────────────────────
function renderMarkdown(s) {
  if (!s) return '';
  let out = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  out = out.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  out = out.replace(/(?<!\\*)\\*([^*\\n]+)\\*(?!\\*)/g, '<em>$1</em>');
  out = out.replace(/^####?\\s+(.+)$/gm, '<h4>$1</h4>');
  const lines = out.split('\\n');
  let html = '';
  let inList = false;
  for (const raw of lines) {
    const m = raw.match(/^\\s*[-•*]\\s+(.+)$/);
    if (m) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${m[1]}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (raw.trim()) html += `<p>${raw}</p>`;
    }
  }
  if (inList) html += '</ul>';
  return html;
}

async function refreshBrief() {
  try {
    const b = await api('/api/market-brief');
    if (!b.available) return;
    const card = document.getElementById('briefCard');
    const meta = document.getElementById('briefMeta');
    const cite = document.getElementById('briefCitations');
    const ts = b.timestamp ? new Date(b.timestamp*1000) : null;
    const tsLabel = ts ? `${ts.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'})} · ${b.session_label} session` : '';
    meta.textContent = tsLabel;
    card.querySelector('.brief-text').classList.remove('skel');
    card.querySelector('.brief-text').innerHTML = renderMarkdown(b.text || '');
    if (b.citations && b.citations.length) {
      cite.innerHTML = 'Sources · ' + b.citations.slice(0, 6).map((c, i) =>
        `<a href="${c}" target="_blank" rel="noopener">[${i+1}]</a>`).join('');
    } else {
      cite.innerHTML = '';
    }
  } catch (e) { console.error('brief', e); }
}

// ── Daily review ───────────────────────────────────────────────
async function refreshReview() {
  try {
    const r = await api('/api/daily-review');
    const section = document.getElementById('reviewSection');
    if (!r.available) { section.style.display = 'none'; return; }
    section.style.display = 'block';

    // Use the first sentence (or first 90 chars) as the headline; the rest is the body.
    const text = (r.text || '').trim();
    const splitIdx = text.search(/[.!?]\\s/);
    const headline = splitIdx > 0 && splitIdx < 140 ? text.slice(0, splitIdx + 1) : text.slice(0, 140);
    const body = splitIdx > 0 ? text.slice(splitIdx + 2) : '';

    document.getElementById('reviewHeadline').textContent = headline;
    const paragraphs = body.split(/\\n\\n+/).filter(p => p.trim());
    document.getElementById('reviewBody').innerHTML = paragraphs.map(p => `<p>${p.trim()}</p>`).join('');

    const dateLabel = r.review_date || '';
    const pnlLabel = (r.day_pnl >= 0 ? '+' : '−') + '$' + Math.abs(r.day_pnl || 0).toFixed(2);
    const cls = (r.day_pnl >= 0) ? 'positive' : '';
    document.getElementById('reviewMeta').innerHTML =
      `${dateLabel}  ·  ${r.n_closed || 0} closed trades  ·  Day P&L <span class="${cls}">${pnlLabel}</span>`;
  } catch (e) { console.error('review', e); }
}

async function refreshAll() {
  try {
    renderSessions();
    await Promise.all([
      refreshStatus(), refreshEquity(), refreshScoreboard(),
      refreshPositions(), refreshDecisions(), refreshSkips(), refreshTrades(),
      refreshBrief(), refreshReview(),
    ]);
  } catch (e) { console.error(e); }
}

refreshAll();
setInterval(refreshAll, 30000);
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    _check_token(request)
    return HTMLResponse(content=HTML)


def serve():
    import uvicorn
    uvicorn.run(
        "velox_edge.dashboard:app",
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    serve()
