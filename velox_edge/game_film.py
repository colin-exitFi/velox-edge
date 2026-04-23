"""Game Film — multi-day pattern recognition + recommendations.

Inherited intent from velox-classic's game_film.py. Critical fix vs the
original: this version analyzes data that's actually been tagged correctly
at decision time. Every trade carries its consensus_confidence, claude_vote,
gpt_vote, side, source category, and exit reason — so the aggregations mean
what they say they mean.

Two modes:
  • Hourly cumulative — runs every 60 min during market hours, refreshes
    the rolling stats so the dashboard always shows current state.
  • Daily report (post-close) — at 16:10 ET, writes a structured report
    with recommendations and persists it. Surfaced on the dashboard as
    "What the data says" beneath the daily review.

Read-only on day 1 — recommendations are observations, not auto-applied.
The 'apply' switch comes after 30 days of trust-building data.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from velox_edge import config, state
from velox_edge.universe import CATEGORY_OF


# Minimum trades required to make a recommendation about something
MIN_TRADES_FOR_INSIGHT = 5


# ── Aggregations ──────────────────────────────────────────────────


def _bucket(trades: List[Dict], key_fn) -> Dict[str, Dict]:
    """Group trades by key_fn and compute per-bucket stats."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pct_pnl_sum": 0.0})
    for t in trades:
        k = key_fn(t)
        if k is None:
            continue
        b = buckets[k]
        b["trades"] += 1
        pnl = float(t.get("pnl") or 0)
        b["pnl"] += pnl
        b["pct_pnl_sum"] += float(t.get("pnl_pct") or 0)
        if pnl > 0:
            b["wins"] += 1
    out = {}
    for k, b in buckets.items():
        n = b["trades"]
        out[k] = {
            "trades": n,
            "wins": b["wins"],
            "win_rate": round(b["wins"] / max(1, n) * 100, 1),
            "total_pnl": round(b["pnl"], 2),
            "avg_pnl": round(b["pnl"] / max(1, n), 2),
            "avg_pct": round(b["pct_pnl_sum"] / max(1, n), 2),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["total_pnl"]))


def _confidence_band(t: Dict) -> str:
    c = float(t.get("consensus_confidence") or 0)
    if c < 60:  return "<60% (off-policy)"
    if c < 70:  return "60-69%"
    if c < 80:  return "70-79%"
    if c < 90:  return "80-89%"
    return "90%+"


def _hold_bucket(t: Dict) -> str:
    s = float(t.get("hold_seconds") or 0)
    m = s / 60
    if m < 5:    return "<5 min"
    if m < 30:   return "5-30 min"
    if m < 120:  return "30 min-2h"
    if m < 240:  return "2-4h"
    return ">4h"


def _category(t: Dict) -> str:
    return CATEGORY_OF.get(t.get("symbol", ""), "scanner_or_other")


def _hour_of_day_et(t: Dict) -> Optional[str]:
    et = t.get("entry_time") or t.get("exit_time")
    if not et:
        return None
    try:
        # Local timestamps from SQLite are unix epoch UTC
        dt = datetime.fromtimestamp(float(et))
        return f"{dt.hour:02d}:00"
    except Exception:
        return None


def _consensus_pattern(t: Dict) -> str:
    """e.g. 'Both BUY', 'Both SHORT', 'Claude BUY / GPT SHORT (we wouldn't have entered)'.
    For closed trades, both votes were on the entered side, so this is mostly:
    'Both BUY' or 'Both SHORT'."""
    cv = (t.get("claude_vote") or "?@0").split("@")[0]
    gv = (t.get("gpt_vote") or "?@0").split("@")[0]
    if cv == gv:
        return f"Both {cv}"
    return f"Claude {cv} / GPT {gv}"


# ── Recommendations ────────────────────────────────────────────────


def _generate_recommendations(stats: Dict) -> List[Dict]:
    """Return a list of human-readable recommendation rows."""
    recs: List[Dict] = []

    # Confidence band — find the threshold below which expectancy goes negative
    bands = stats.get("by_confidence", {})
    for label, b in bands.items():
        if b["trades"] < MIN_TRADES_FOR_INSIGHT:
            continue
        if b["avg_pnl"] < 0 and "60-69%" in label:
            recs.append({
                "kind": "raise_min_confidence",
                "severity": "warn",
                "summary": f"Trades at 60-69% confidence are losing on average (${b['avg_pnl']:.2f} per trade over {b['trades']} trades). Consider raising MIN_CONSENSUS_CONFIDENCE to 70.",
            })

    # Worst categories — at least N trades, negative expectancy
    cats = stats.get("by_category", {})
    losing = [(k, v) for k, v in cats.items() if v["trades"] >= MIN_TRADES_FOR_INSIGHT and v["avg_pnl"] < 0]
    losing.sort(key=lambda kv: kv[1]["avg_pnl"])
    for cat, v in losing[:3]:
        recs.append({
            "kind": "category_warning",
            "severity": "warn",
            "summary": f"Category '{cat}' is losing: ${v['total_pnl']:.2f} over {v['trades']} trades ({v['win_rate']:.0f}% WR). Consider tightening or pausing this bucket.",
        })

    # Best categories — affirmation
    winning = [(k, v) for k, v in cats.items() if v["trades"] >= MIN_TRADES_FOR_INSIGHT and v["avg_pnl"] > 0]
    winning.sort(key=lambda kv: -kv[1]["avg_pnl"])
    for cat, v in winning[:2]:
        recs.append({
            "kind": "category_strength",
            "severity": "info",
            "summary": f"Category '{cat}' is your edge: ${v['total_pnl']:.2f} over {v['trades']} trades ({v['win_rate']:.0f}% WR, ${v['avg_pnl']:.2f}/trade). Keep doing this.",
        })

    # Hold-time pattern
    holds = stats.get("by_hold_duration", {})
    if holds:
        # Compare avg_pnl by bucket
        sorted_holds = sorted(holds.items(), key=lambda kv: kv[1].get("avg_pnl", 0), reverse=True)
        if len(sorted_holds) >= 2:
            best_label, best = sorted_holds[0]
            worst_label, worst = sorted_holds[-1]
            if best["trades"] >= MIN_TRADES_FOR_INSIGHT and worst["trades"] >= MIN_TRADES_FOR_INSIGHT:
                if best["avg_pnl"] > 0 and worst["avg_pnl"] < 0 and abs(best["avg_pnl"] - worst["avg_pnl"]) > 1.0:
                    recs.append({
                        "kind": "hold_time_pattern",
                        "severity": "info",
                        "summary": f"Hold-time edge: '{best_label}' trades average ${best['avg_pnl']:.2f}, '{worst_label}' trades average ${worst['avg_pnl']:.2f}. Tighten ratchet to favor {best_label}.",
                    })

    # Symbol blacklist candidates — losing on multiple attempts
    syms = stats.get("by_symbol", {})
    chronic_losers = [(k, v) for k, v in syms.items() if v["trades"] >= 3 and v["total_pnl"] < -10]
    chronic_losers.sort(key=lambda kv: kv[1]["total_pnl"])
    for sym, v in chronic_losers[:5]:
        recs.append({
            "kind": "symbol_blacklist",
            "severity": "warn",
            "summary": f"{sym}: {v['trades']} trades, ${v['total_pnl']:.2f} total, {v['win_rate']:.0f}% WR. Consider removing from universe or shadow-only.",
        })

    # Confidence-band sizing recommendation
    high = bands.get("80-89%", {"trades": 0, "avg_pnl": 0, "win_rate": 0})
    very_high = bands.get("90%+", {"trades": 0, "avg_pnl": 0, "win_rate": 0})
    if (high["trades"] + very_high["trades"]) >= MIN_TRADES_FOR_INSIGHT:
        if (high["avg_pnl"] + very_high["avg_pnl"]) / 2 > 1.0:
            recs.append({
                "kind": "lean_into_conviction",
                "severity": "info",
                "summary": f"High-confidence trades (80%+) are paying — consider raising POSITION_SIZE_MAX_PCT to lean into them harder.",
            })

    # Claude vs GPT patterns (when they agreed and we entered)
    consensus = stats.get("by_consensus_pattern", {})
    if consensus.get("Both BUY", {}).get("trades", 0) >= MIN_TRADES_FOR_INSIGHT:
        bb = consensus["Both BUY"]
        if bb["avg_pnl"] < 0:
            recs.append({
                "kind": "long_bias_check",
                "severity": "warn",
                "summary": f"'Both BUY' consensus trades losing on avg (${bb['avg_pnl']:.2f}/trade over {bb['trades']}). Check if regime favors shorts.",
            })

    return recs


# ── Main run ──────────────────────────────────────────────────────


def compute_game_film(lookback_days: int = 30) -> Dict:
    """Compute game film stats from the trade DB. Pure function — no side effects.

    Returns: {
        meta: {trade_count, lookback_days, generated_at, ...},
        overall: {trades, win_rate, total_pnl, ...},
        by_category, by_symbol, by_confidence, by_hold_duration,
        by_hour_of_day, by_consensus_pattern, by_exit_reason,
        recommendations: [...],
    }
    """
    cutoff = time.time() - (lookback_days * 24 * 3600)
    all_closed = state.recent_closed_trades(limit=1000)
    trades = [t for t in all_closed if (t.get("exit_time") or 0) >= cutoff]

    if not trades:
        return {
            "meta": {
                "trade_count": 0,
                "lookback_days": lookback_days,
                "generated_at": time.time(),
                "min_required": MIN_TRADES_FOR_INSIGHT,
                "ready": False,
                "message": "No closed trades yet — game film waits for data.",
            },
        }

    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) <= 0]

    overall = {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / max(1, len(trades)) * 100, 1),
        "total_pnl": round(sum((t.get("pnl") or 0) for t in trades), 2),
        "avg_winner": round(sum((t.get("pnl") or 0) for t in wins) / max(1, len(wins)), 2),
        "avg_loser": round(sum((t.get("pnl") or 0) for t in losses) / max(1, len(losses)), 2),
        "avg_pct": round(sum((t.get("pnl_pct") or 0) for t in trades) / max(1, len(trades)), 2),
        "avg_winner_hold_min": round(
            sum((t.get("hold_seconds") or 0) for t in wins) / max(1, len(wins)) / 60, 1
        ),
        "avg_loser_hold_min": round(
            sum((t.get("hold_seconds") or 0) for t in losses) / max(1, len(losses)) / 60, 1
        ),
    }

    stats = {
        "by_symbol":            _bucket(trades, lambda t: t.get("symbol")),
        "by_category":          _bucket(trades, _category),
        "by_confidence":        _bucket(trades, _confidence_band),
        "by_hold_duration":     _bucket(trades, _hold_bucket),
        "by_hour_of_day":       _bucket(trades, _hour_of_day_et),
        "by_consensus_pattern": _bucket(trades, _consensus_pattern),
        "by_exit_reason":       _bucket(trades, lambda t: t.get("exit_reason") or "unknown"),
        "by_side":              _bucket(trades, lambda t: t.get("side")),
    }

    recs = _generate_recommendations(stats)

    return {
        "meta": {
            "trade_count": len(trades),
            "lookback_days": lookback_days,
            "generated_at": time.time(),
            "min_required": MIN_TRADES_FOR_INSIGHT,
            "ready": len(trades) >= MIN_TRADES_FOR_INSIGHT,
            "message": (
                f"Game film: {len(trades)} closed trades over last {lookback_days} days. "
                f"{len(recs)} observations."
            ),
        },
        "overall": overall,
        **stats,
        "recommendations": recs,
    }


def write_game_film() -> Optional[Dict]:
    """Compute + persist a snapshot. Called by main.py at 16:10 ET."""
    insights = compute_game_film(lookback_days=30)
    state.record_game_film(insights)
    logger.info(
        f"🎬 Game film: {insights['meta'].get('trade_count', 0)} closed trades in window, "
        f"{len(insights.get('recommendations', []))} recommendations"
    )
    return insights
