"""Daily review — Claude reads the day's data and writes an editorial summary.

Inherited learning from velox-classic + ARC's post-mortem: the recursive
self-review loop is what makes the system get better over time. Not a journal
entry — a structured retrospective that should surface patterns Velox won't
notice in real time.

Runs once per trading day at 16:00 ET (after pre-close session, before EOD
flatten finalizes). Costs ~$0.10/day in Claude tokens. Output is persisted
and displayed on the dashboard as 'Today, in 200 words'.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from loguru import logger

from velox_edge import config, state


_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


PROMPT_TEMPLATE = """You are the chief reviewer for Velox Edge, a paper trading bot.
The bot ran today using a Perplexity market brief + Claude+GPT consensus voting +
a profit ratchet. You're writing a tight, structured retrospective an operator
will actually read.

═══ TODAY ═══
Date: {date}
Sessions completed: {n_sessions} of 5
Decisions evaluated: {n_decisions}
Trades opened: {n_opened}
Trades closed: {n_closed}
Trades skipped on disagreement: {n_disagree}
Trades skipped on low conviction: {n_lowconf}
Trades skipped on concentration cap: {n_concentration}

═══ TODAY'S MARKET BRIEF (first session) ═══
{first_brief}

═══ CLOSED TRADES ═══
{closed_trades_block}

═══ NOTABLE SKIPS (where Claude and GPT disagreed) ═══
{notable_skips_block}

═══ EQUITY ═══
Starting: ${start_eq:,.2f}
Ending:   ${end_eq:,.2f}
Day P&L:  ${day_pnl:+,.2f} ({day_pct:+.2f}%)

═══ YOUR TASK ═══
Write a tight retrospective of today (max 250 words, prose, no markdown
headings). Cover, in this order:

1. What happened — one paragraph framing the day's action.
2. Where the consensus filter EARNED its keep — name 1-2 specific skips
   that look correct in hindsight (a model wanted in, the other was right
   to disagree). Cite tickers.
3. Where the consensus filter PROBABLY COST us — name 1-2 skips that
   look wrong in hindsight (we left money on the table). Cite tickers.
4. Pattern observation — is there a regime, sector, or model bias forming
   in the data? Be specific about what to watch tomorrow.
5. One concrete adjustment to consider for tomorrow's first session —
   smaller, sharper, less noisy than 'be smarter.'

Style: a senior PM writing to themselves at end of day. Direct, honest,
no padding. Skip section labels — flow as paragraphs."""


def _format_trades(trades: List[Dict]) -> str:
    if not trades:
        return "(no trades closed today)"
    lines = []
    for t in trades:
        side = t.get("side", "?")
        sym = t.get("symbol", "?")
        ep = t.get("entry_price") or 0
        xp = t.get("exit_price") or 0
        pnl = t.get("pnl") or 0
        pnl_pct = t.get("pnl_pct") or 0
        hold_min = round((t.get("hold_seconds") or 0) / 60)
        reason = t.get("exit_reason") or ""
        cv = t.get("claude_vote") or "?"
        gv = t.get("gpt_vote") or "?"
        lines.append(
            f"  {sym:<6} {side:<5} ${ep:.2f}→${xp:.2f}  pnl={pnl:+.2f} ({pnl_pct:+.2f}%) "
            f"held {hold_min}m  exit={reason}  votes[C:{cv} G:{gv}]"
        )
    return "\n".join(lines)


def _format_skips(skips: List[Dict], limit: int = 8) -> str:
    if not skips:
        return "(no notable disagreements today)"
    notable = [s for s in skips if "disagreement" in (s.get("skip_reason") or "")][:limit]
    if not notable:
        return "(no model disagreements today)"
    lines = []
    for s in notable:
        sym = s.get("symbol", "?")
        ca = s.get("claude_action", "?")
        cc = s.get("claude_confidence", 0)
        ga = s.get("gpt_action", "?")
        gc = s.get("gpt_confidence", 0)
        lines.append(f"  {sym:<6} Claude:{ca}@{cc:.0f}  GPT:{ga}@{gc:.0f}  → SKIPPED")
    return "\n".join(lines)


async def write_daily_review() -> Optional[str]:
    """Generate today's review and persist it. Returns the review text."""
    if not config.ANTHROPIC_API_KEY or not config.DAILY_REVIEW_ENABLED:
        logger.info("Daily review disabled — skipping")
        return None

    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    # Pull today's data
    decisions = [d for d in state.recent_decisions(limit=500) if d["timestamp"] >= today_start]
    skips = [d for d in decisions if not d["executed"] and d.get("skip_reason")]
    closed = [t for t in state.recent_closed_trades(limit=200) if (t.get("exit_time") or 0) >= today_start]
    briefs = [b for b in state.recent_market_briefs(limit=10) if b["timestamp"] >= today_start]

    sessions_today = len(set(d.get("session_id") for d in decisions if d.get("session_id")))
    n_decisions = len(decisions)
    n_opened = sum(1 for d in decisions if d["executed"] and d["consensus_action"] in ("BUY", "SHORT"))
    n_closed = len(closed)
    n_disagree = sum(1 for s in skips if "disagreement" in (s.get("skip_reason") or ""))
    n_lowconf = sum(1 for s in skips if "low_conviction" in (s.get("skip_reason") or ""))
    n_concentration = sum(1 for s in skips if "concentration_cap" in (s.get("skip_reason") or ""))

    # Equity bookends from equity_history
    history = state.equity_history()
    today_eq = [pt for pt in history if pt.get("timestamp", 0) >= today_start]
    if today_eq:
        start_eq = today_eq[0]["equity"]
        end_eq = today_eq[-1]["equity"]
    else:
        start_eq = end_eq = 0.0
    day_pnl = end_eq - start_eq
    day_pct = (day_pnl / start_eq * 100) if start_eq else 0.0

    first_brief = briefs[-1]["text"] if briefs else "(no brief recorded today)"

    prompt = PROMPT_TEMPLATE.format(
        date=datetime.now().strftime("%A, %B %d %Y"),
        n_sessions=sessions_today,
        n_decisions=n_decisions,
        n_opened=n_opened,
        n_closed=n_closed,
        n_disagree=n_disagree,
        n_lowconf=n_lowconf,
        n_concentration=n_concentration,
        first_brief=first_brief[:1500],
        closed_trades_block=_format_trades(closed),
        notable_skips_block=_format_skips(skips),
        start_eq=start_eq,
        end_eq=end_eq,
        day_pnl=day_pnl,
        day_pct=day_pct,
    )

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": config.ANTHROPIC_MODEL,
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"].strip()
    except Exception as e:
        logger.error(f"Daily review failed: {e}")
        return None

    logger.info(
        f"📝 Daily review written ({len(text)} chars in {time.time()-started:.1f}s) — "
        f"sessions={sessions_today} closed={n_closed} pnl=${day_pnl:+.2f}"
    )
    state.record_daily_review(text=text, day_pnl=day_pnl, n_closed=n_closed)
    return text
