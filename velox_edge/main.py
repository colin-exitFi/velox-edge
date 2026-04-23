"""Velox Edge main loop.

Schedule:
  - 5 consensus sessions per trading day at 9:35, 11:00, 13:30, 15:00, 15:45 ET
  - Between sessions: 30s ratchet tick on open positions
  - 15:55 ET: force-close all positions (no overnight in v1)

All durable state lives in SQLite (data/velox_edge.db). Equity history is
appended to data/equity_history.json after every session.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from loguru import logger

from velox_edge import (
    broker, config, consensus, market_brief, ratchet, review,
    scanner, sizing, state,
)
from velox_edge.universe import UNIVERSE


ET = pytz.timezone("America/New_York")

# Edge's cadence is sparser — 8 sessions/day. The model is looking for
# clear contrarian extremes; firing every 30 min produces noise on a small
# universe. Sessions skew toward open and close where extremes live.
SESSIONS: List[Tuple[int, int, str]] = [
    (9, 40,  "open_settle"),     # 5 min after open, let spreads tighten
    (10, 15, "morning_drift"),
    (11, 0,  "mid_morning"),
    (12, 30, "lunch_reversal"),
    (14, 0,  "afternoon_setup"),
    (15, 0,  "power_hour"),
    (15, 30, "late_fade"),
    (15, 50, "pre_close"),
]

EOD_FLATTEN = (15, 55)  # only fires if EDGE_FORCE_FLATTEN_AT_EOD=true
RATCHET_TICK_SECONDS = 60  # less frequent — wider stops mean fewer false alarms


def _now_et() -> datetime:
    return datetime.now(pytz.utc).astimezone(ET)


def _next_session_et(now: Optional[datetime] = None) -> Tuple[datetime, str]:
    now = now or _now_et()
    today = now.date()
    candidates = []
    for h, m, label in SESSIONS:
        t = ET.localize(datetime.combine(today, dtime(h, m)))
        if t > now:
            candidates.append((t, label))
    if candidates:
        return candidates[0]
    # All today's sessions are past — schedule next trading day's first session
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:  # skip weekends; broker calendar handles holidays
        next_day += timedelta(days=1)
    h, m, label = SESSIONS[0]
    return ET.localize(datetime.combine(next_day, dtime(h, m))), label


def _eod_flatten_today(now: Optional[datetime] = None) -> datetime:
    now = now or _now_et()
    h, m = EOD_FLATTEN
    return ET.localize(datetime.combine(now.date(), dtime(h, m)))


# ── Session: gather data, ask AI, place orders ─────────────────────


async def run_session(label: str):
    logger.info(f"=== SESSION: {label} ===")
    state.audit("session_start", "info", label)

    if config.TRADING_HALTED:
        logger.warning("TRADING_HALTED=true — skipping session")
        state.audit("session_skipped", "warn", "trading_halted")
        return

    if not await broker.is_market_open():
        logger.info("Market is closed — skipping session")
        state.audit("session_skipped", "info", "market_closed")
        return

    equity = await broker.get_equity()

    # Edge runs almost entirely on the scanner — small ETF anchor + 30+ scanner picks.
    # Bigger pool than vanilla because contrarian setups are rarer per-name.
    scan = await scanner.daily_scan(target_count=35)
    full_universe = list(UNIVERSE) + list(scan["tickers"])
    snaps = await broker.get_snapshots(full_universe)
    if not snaps:
        logger.error("No snapshots returned — abort session")
        state.audit("session_failed", "error", "no_snapshots")
        return

    positions = await broker.get_positions()
    open_symbols = [p["symbol"] for p in positions]
    n_open = len(positions)

    session_id = state.start_session(label, equity)
    logger.info(
        f"Session #{session_id} | {label} | equity=${equity:,.2f} | "
        f"universe={len(full_universe)} ({len(UNIVERSE)} anchor + "
        f"{len(scan['tickers'])} scanned) | open={n_open} snapshots={len(snaps)}"
    )

    # 1. Pull market context from Perplexity FIRST so both voters see the same world.
    brief = await market_brief.get_market_brief(full_universe, label)
    state.record_market_brief(
        session_id=session_id,
        session_label=label,
        text=brief.text,
        citations=brief.citations,
        error=brief.error,
    )

    # 2. Two voters now decide with context — both the brief AND the scanner overlay.
    votes_by_symbol = await consensus.run_consensus(
        snapshots=snaps,
        open_positions=open_symbols,
        session_label=label,
        equity=equity,
        max_positions=config.MAX_CONCURRENT_POSITIONS,
        market_brief=brief.text,
        scanner_details=scan["details"],
    )

    opened = 0
    skipped = 0

    for symbol, votes in votes_by_symbol.items():
        snap = snaps.get(symbol) or {}
        price = float(snap.get("price") or 0)
        if price <= 0:
            continue
        cv = votes["claude"]
        gv = votes["gpt"]
        c = votes["consensus"]
        action = c["action"]
        conf = c["confidence"]
        skip_reason = c.get("skip_reason", "")

        # Capacity check for new entries
        already_held = symbol in open_symbols
        if action in ("BUY", "SHORT") and not already_held:
            if n_open >= config.MAX_CONCURRENT_POSITIONS:
                state.record_decision(
                    session_id, symbol, price, cv, gv, "HOLD", conf,
                    skip_reason="position_cap_reached", executed=False,
                )
                skipped += 1
                continue

        executed = False

        if action in ("BUY", "SHORT") and not already_held:
            # Conviction-based sizing inherited from velox-classic learnings.
            qty, intended_pct = sizing.size_position(equity, price, conf)
            if qty > 0:
                # Concentration guard inherited from velox-classic.
                concentration_block = sizing.concentration_block_reason(
                    symbol=symbol,
                    intended_notional=qty * price,
                    equity=equity,
                    positions=positions,
                )
                if concentration_block:
                    state.record_decision(
                        session_id, symbol, price, cv, gv, "HOLD", conf,
                        skip_reason=concentration_block, executed=False,
                    )
                    skipped += 1
                    continue

                side_label = "buy" if action == "BUY" else "sell"
                trade_side = "long" if action == "BUY" else "short"
                order = await broker.submit_market_order(symbol, qty, side_label)
                if order:
                    decision_id = state.record_decision(
                        session_id, symbol, price, cv, gv, action, conf,
                        skip_reason="", executed=True,
                    )
                    state.record_trade_open(
                        decision_id, symbol, trade_side, price, qty, conf, cv, gv
                    )
                    state.audit(
                        "trade_open", "info",
                        f"{symbol} {trade_side} qty={qty} @ ${price:.2f} "
                        f"conf={conf:.0f} size={intended_pct:.2f}%",
                    )
                    n_open += 1
                    opened += 1
                    executed = True

        elif action == "EXIT" and already_held:
            await _close_position_by_symbol(symbol, "consensus_exit")
            state.record_decision(
                session_id, symbol, price, cv, gv, "EXIT", conf,
                skip_reason="", executed=True,
            )
            opened += 1  # counts as an action; not really an "open" but a session action
            executed = True

        if not executed:
            state.record_decision(
                session_id, symbol, price, cv, gv, action, conf,
                skip_reason=skip_reason or ("no_action_required" if action == "HOLD" else ""),
                executed=False,
            )
            if skip_reason:
                skipped += 1

    final_equity = await broker.get_equity()
    spy_price = float((snaps.get("SPY") or {}).get("price") or 0)
    state.append_equity_point(final_equity, spy_price)
    state.end_session(session_id, final_equity, opened, skipped)
    logger.info(
        f"=== SESSION DONE: {label} | opened={opened} skipped={skipped} "
        f"equity=${final_equity:,.2f} ==="
    )
    state.audit(
        "session_end", "info",
        f"{label} opened={opened} skipped={skipped} equity=${final_equity:.2f}",
    )


# ── Ratchet management for open positions ──────────────────────────


_ratchet_states: Dict[str, ratchet.RatchetState] = {}


async def ratchet_tick():
    positions = await broker.get_positions()
    if not positions:
        # purge stale ratchet states
        if _ratchet_states:
            _ratchet_states.clear()
        return

    syms = [p["symbol"] for p in positions]
    snaps = await broker.get_snapshots(syms)

    # Ensure each open position has a ratchet state (rebuilt from broker if needed)
    for p in positions:
        sym = p["symbol"]
        if sym not in _ratchet_states:
            entry_price = float(p.get("avg_entry_price") or 0)
            qty = float(p.get("qty") or 0)
            side = "long" if qty > 0 else "short"
            _ratchet_states[sym] = ratchet.RatchetState(
                entry_price=entry_price,
                entry_time=time.time(),  # best effort; broker doesn't expose entry timestamp
                side=side,
            )

    for sym, rstate in list(_ratchet_states.items()):
        snap = snaps.get(sym)
        if not snap:
            continue
        price = float(snap.get("price") or 0)
        if price <= 0:
            continue
        ratchet.update_peak(rstate, price)
        reason = ratchet.should_exit(rstate, price)
        if reason:
            logger.info(
                f"🔻 RATCHET EXIT {sym}: reason={reason} pnl={ratchet.pnl_pct(rstate, price):.2f}% "
                f"peak={rstate.peak_pnl_pct:.2f}%"
            )
            await _close_position_by_symbol(sym, reason)


async def _close_position_by_symbol(symbol: str, reason: str):
    """Close a position via broker and record the exit in DB."""
    snap = await broker.get_snapshots([symbol])
    exit_price = float((snap.get(symbol) or {}).get("price", 0)) or 0
    res = await broker.close_position(symbol)
    if res is not None:
        trade = state.find_open_trade(symbol)
        if trade and exit_price > 0:
            state.record_trade_close(trade["id"], exit_price, reason)
        state.audit("trade_close", "info", f"{symbol} reason={reason} price=${exit_price:.2f}")
        _ratchet_states.pop(symbol, None)


# ── End-of-day flatten ─────────────────────────────────────────────


async def flatten_all(reason: str = "eod_flatten"):
    positions = await broker.get_positions()
    for p in positions:
        await _close_position_by_symbol(p["symbol"], reason)
    state.audit("flatten_all", "info", f"reason={reason} closed={len(positions)}")


async def _flatten_stale_positions(max_days: int):
    """Force-close any position held longer than max_days. Edge keeps winners
    overnight but caps the hold so a thesis can't drift into purgatory.
    """
    open_trades = state.get_open_trades()
    cutoff = time.time() - (max_days * 24 * 3600)
    stale = [t for t in open_trades if (t.get("entry_time") or 0) < cutoff]
    for t in stale:
        sym = t.get("symbol")
        if sym:
            logger.info(f"⏳ Stale position close: {sym} held > {max_days}d")
            await _close_position_by_symbol(sym, f"max_hold_{max_days}d")


# ── Main loop orchestration ────────────────────────────────────────


_executed_today: set = set()  # session keys (date_label) we've run already


def _session_key(now: datetime, label: str) -> str:
    return f"{now.date().isoformat()}_{label}"


async def main_loop():
    logger.info("⚡ Velox Edge starting")
    state.init_db()
    state.audit("startup", "info", "velox-core booted")

    # Sanity check Alpaca connectivity
    try:
        acct = await broker.get_account()
        equity = float(acct.get("equity", 0))
        logger.success(
            f"Alpaca connected (paper={config.ALPACA_PAPER}) | equity=${equity:,.2f}"
        )
    except Exception as e:
        logger.error(f"Alpaca init failed: {e}")
        state.audit("startup_error", "error", f"alpaca_init: {e}")
        sys.exit(1)

    while True:
        try:
            await _tick()
        except Exception as e:
            logger.error(f"main loop error: {e}")
            state.audit("main_loop_error", "error", str(e))
        await asyncio.sleep(RATCHET_TICK_SECONDS)


async def _tick():
    now = _now_et()

    # 1. EOD flatten check — Edge can hold overnight by default, so this only
    # fires if EDGE_FORCE_FLATTEN_AT_EOD=true OR a position has hit its max hold.
    eod = _eod_flatten_today(now)
    eod_key = f"{now.date().isoformat()}_eod_flatten"
    if now >= eod and eod_key not in _executed_today and await broker.is_market_open():
        if config.EDGE_FORCE_FLATTEN_AT_EOD:
            logger.info("⏰ EOD flatten triggered (EDGE_FORCE_FLATTEN_AT_EOD=true)")
            await flatten_all("eod_flatten")
        else:
            # Selective flatten: only positions held > MAX_HOLD_DAYS
            await _flatten_stale_positions(max_days=config.EDGE_MAX_HOLD_DAYS)
        _executed_today.add(eod_key)

    # 2. Session check — fire any session whose scheduled time has passed today
    for h, m, label in SESSIONS:
        scheduled = ET.localize(datetime.combine(now.date(), dtime(h, m)))
        key = _session_key(now, label)
        if now >= scheduled and key not in _executed_today:
            await run_session(label)
            _executed_today.add(key)

    # 3. Daily review — fires once per trading day at the configured ET time
    review_dt = ET.localize(
        datetime.combine(
            now.date(),
            dtime(config.DAILY_REVIEW_HOUR_ET, config.DAILY_REVIEW_MIN_ET),
        )
    )
    review_key = f"{now.date().isoformat()}_daily_review"
    if (
        config.DAILY_REVIEW_ENABLED
        and now >= review_dt
        and review_key not in _executed_today
        and now.weekday() < 5  # weekdays only
    ):
        try:
            logger.info("📝 Writing daily review…")
            await review.write_daily_review()
        except Exception as e:
            logger.error(f"Daily review error: {e}")
            state.audit("daily_review_error", "error", str(e))
        _executed_today.add(review_key)

    # 4. Ratchet on open positions (every tick during market hours)
    if await broker.is_market_open():
        await ratchet_tick()

    # Reset daily marker set at midnight
    if now.hour == 0 and now.minute < 5:
        _executed_today.clear()


# ── Entrypoint ─────────────────────────────────────────────────────


def _setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


if __name__ == "__main__":
    _setup_logging()
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
