"""Claude vs GPT consensus engine.

One call per model per session. Each model receives the FULL universe snapshot
in a single prompt and returns a JSON list of {symbol, action, confidence, reason}
for every ticker. We then merge votes — entries fire only on agreement at
≥MIN_CONSENSUS_CONFIDENCE.

Cost discipline:
 - 1 Claude call + 1 GPT call per session × 5 sessions/day = 10 calls/day
 - ~3-5K input tokens per call, ~2-3K output tokens
 - At Claude Sonnet + GPT-5.4-mini pricing: ~$1-2/day, ~$30-50/month
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

import httpx
from loguru import logger

from velox_edge import config


_TIMEOUT = httpx.Timeout(45.0, connect=10.0)
_VALID_ACTIONS = {"BUY", "SHORT", "HOLD", "EXIT"}


PROMPT_TEMPLATE = """You are a CONTRARIAN trading agent inside Velox Edge, an autonomous paper trading bot.

YOUR MANDATE IS DIFFERENT FROM A TYPICAL MOMENTUM BOT.
Your job is to find where the crowd is wrong and take the other side.

The literature you stand on:
  • Stocks up >100% on day 1 → median return -12% over next 5 days
  • Top-decile single-day gainers underperform top-decile losers over 30 days
  • Retail capitulation lows reverse 70%+ of the time on liquid names
  • The most-active list at midday is where dumb money is concentrated

Your job: score each of the {n} stocks below and return a contrarian action.

CURRENT TIME: {now} ET ({session_label})
EQUITY: ${equity:,.0f}
OPEN POSITIONS: {open_positions}

Already held ({n_open} total): {position_list}

═══ MARKET BRIEF (Perplexity, fresh) ═══
{market_brief}
═══════════════════════════════════════════════════════════

═══ OPTIONS FLOW (Unusual Whales, last 4h on scanner candidates) ═══
{options_flow_block}

How to read this:
  • >60% calls + 'mostly retail' → classic dumb-money pile, strong fade signal
  • >60% calls + 'institutional sweeps' → smart money buying, DON'T fade
  • >60% puts + 'institutional sweeps' → smart money positioning short, lean SHORT
  • Balanced or low total premium → no edge from flow, decide on technicals
═══════════════════════════════════════════════════════════════════════

Universe snapshot — almost ALL of these are today's most-moved names. That's
the whole point. They're here BECAUSE the crowd is on them right now. Your
job is to look for the ones where the crowd is most likely wrong.

{universe_table}

For EACH symbol return one of:
  SHORT  — fade the move; conviction the price reverts in next 30-180 min OR over 1-3 days
  BUY    — buy the dip / capitulation bounce; conviction price is washed out and reverts up
  HOLD   — no clear extreme either way
  EXIT   — close an existing position now (only for symbols already held)

CONTRARIAN HEURISTICS (use these explicitly):
  • A name up 50%+ today on no real catalyst (per the brief) → strong SHORT candidate
  • A name up 20-40% today after a multi-day run → fade the parabola SHORT
  • A name down 30%+ today on a known catalyst that's already priced in → BUY the capitulation
  • A name halted multiple times today → AVOID, too much squeeze risk both ways
  • A name with a NAMED hard catalyst in the brief (FDA, M&A, earnings beat) → don't fade, the move is real
  • SPY/QQQ extreme intraday moves rarely reverse same-day — usually HOLD on indices unless brief shows clear regime shift

RULES:
- The brief is your filter for "is this catalyst real or noise?" If real, don't fade.
- Confidence below 65 = HOLD (we use a higher bar than vanilla momentum).
- Reason: one tight sentence. Name the contrarian setup, e.g. "+187% on stale news, exhaustion candle, fade".
- We only trade consensus — your vote alone fires nothing. The other model sees identical brief + snapshot.
- The ratchet uses WIDER stops here (-3% hard stop, +1% activation) — squeeze risk is real on these names.
- Max concurrent positions: {max_positions}. Sizing is conviction-based 5-10%.

Return ONLY valid JSON in this exact shape (no markdown, no preamble):
{{"votes": [
  {{"symbol": "ATAI", "action": "SHORT", "confidence": 78, "reason": "+187% on stale FDA news per brief, parabolic exhaustion"}},
  {{"symbol": "RIVN", "action": "BUY", "confidence": 67, "reason": "down 32% on guidance, oversold, gap-fill setup"}},
  {{"symbol": "SPY", "action": "HOLD", "confidence": 50, "reason": "no contrarian signal on index"}}
]}}

You MUST return one entry per symbol in the universe ({n} entries total)."""


def _build_universe_table(snapshots: Dict[str, Dict]) -> str:
    rows = []
    for sym, snap in snapshots.items():
        if not snap or not snap.get("price"):
            continue
        vol_ratio = (snap["volume"] / snap["prev_volume"]) if snap.get("prev_volume") else 0
        rows.append(
            f"  {sym:<6} ${snap['price']:>8.2f}  {snap['change_pct']:>+6.2f}%  "
            f"vol={snap['volume']:>10,}  vol_ratio={vol_ratio:>4.1f}x  "
            f"vwap=${snap.get('vwap', 0):>8.2f}"
        )
    return "\n".join(rows)


def _token_budget_for_universe_size(n: int) -> int:
    """Scale output budget with universe size. 120 tokens per ticker + buffer."""
    return max(4000, min(16000, n * 120 + 400))


async def _call_claude(prompt: str, token_budget: int = 4000) -> Optional[Dict]:
    if not config.ANTHROPIC_API_KEY:
        return None
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
                    "max_tokens": token_budget,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"]
            logger.info(f"Claude responded in {time.time()-started:.1f}s ({len(text)} chars, budget={token_budget})")
            return _parse_json(text)
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        logger.error(f"Claude call failed: HTTP {e.response.status_code if e.response else '?'} body={body!r}")
        return None
    except Exception as e:
        logger.error(f"Claude call failed: {type(e).__name__}: {e!r}")
        return None


async def _call_gpt(prompt: str, token_budget: int = 4000) -> Optional[Dict]:
    if not config.OPENAI_API_KEY:
        return None
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.OPENAI_MODEL,
                    "max_completion_tokens": token_budget,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            logger.info(f"GPT responded in {time.time()-started:.1f}s ({len(text)} chars, budget={token_budget})")
            return _parse_json(text)
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        logger.error(f"GPT call failed: HTTP {e.response.status_code if e.response else '?'} body={body!r}")
        return None
    except Exception as e:
        logger.error(f"GPT call failed: {type(e).__name__}: {e!r}")
        return None


def _parse_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find the JSON object inside
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _vote_map(payload: Optional[Dict]) -> Dict[str, Dict]:
    """Convert {votes: [...]} into {symbol: {action, confidence, reason}}."""
    out: Dict[str, Dict] = {}
    if not payload:
        return out
    for entry in payload.get("votes") or []:
        sym = str(entry.get("symbol", "")).upper().strip()
        action = str(entry.get("action", "HOLD")).upper().strip()
        if action not in _VALID_ACTIONS:
            action = "HOLD"
        try:
            conf = float(entry.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(entry.get("reason", ""))[:200]
        if sym:
            out[sym] = {"action": action, "confidence": conf, "reason": reason}
    return out


def consensus_for_symbol(claude_vote: Dict, gpt_vote: Dict) -> Dict:
    """Merge two votes into one consensus action.

    Rules:
      - If either model is missing/HOLD → HOLD with reason capturing why
      - If actions agree (BUY/BUY, SHORT/SHORT, EXIT/EXIT) AND both confidences
        ≥ MIN_CONSENSUS_CONFIDENCE → consensus_action = that action,
        consensus_confidence = mean of the two
      - If actions disagree (e.g. BUY vs SHORT, BUY vs HOLD) → HOLD with skip_reason
      - If actions agree but confidence < threshold → HOLD with skip_reason
    """
    c_action = (claude_vote or {}).get("action", "HOLD")
    g_action = (gpt_vote or {}).get("action", "HOLD")
    c_conf = float((claude_vote or {}).get("confidence", 0))
    g_conf = float((gpt_vote or {}).get("confidence", 0))

    if not claude_vote and not gpt_vote:
        return {"action": "HOLD", "confidence": 0, "skip_reason": "both_models_unavailable"}
    if not claude_vote:
        return {"action": "HOLD", "confidence": 0, "skip_reason": "claude_unavailable"}
    if not gpt_vote:
        return {"action": "HOLD", "confidence": 0, "skip_reason": "gpt_unavailable"}

    if c_action != g_action:
        return {
            "action": "HOLD",
            "confidence": (c_conf + g_conf) / 2,
            "skip_reason": f"disagreement_{c_action}_vs_{g_action}",
        }

    # Actions agree
    if c_action == "HOLD":
        return {"action": "HOLD", "confidence": (c_conf + g_conf) / 2, "skip_reason": ""}

    avg_conf = (c_conf + g_conf) / 2
    if avg_conf < config.MIN_CONSENSUS_CONFIDENCE:
        return {
            "action": "HOLD",
            "confidence": avg_conf,
            "skip_reason": f"low_conviction_{avg_conf:.0f}_below_{config.MIN_CONSENSUS_CONFIDENCE:.0f}",
        }

    return {"action": c_action, "confidence": avg_conf, "skip_reason": ""}


async def run_consensus(
    snapshots: Dict[str, Dict],
    open_positions: List[str],
    session_label: str,
    equity: float,
    max_positions: int,
    market_brief: str = "",
    scanner_details: Optional[List[Dict]] = None,
    options_flow_block: str = "",
) -> Dict[str, Dict]:
    """Call both models, return per-symbol consensus dict.

    Returned shape:
      {symbol: {
          'claude': {action, confidence, reason},
          'gpt': {action, confidence, reason},
          'consensus': {action, confidence, skip_reason}
      }}
    """
    from datetime import datetime
    import asyncio

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    universe_table = _build_universe_table(snapshots)
    # Append a callout for scanner-added names so models weight catalysts harder.
    if scanner_details:
        scanner_lines = []
        for d in scanner_details:
            sym = d.get("symbol", "")
            src = d.get("source", "")
            extras = []
            if "pct_change" in d: extras.append(f"{d['pct_change']:+.2f}%")
            if "volume" in d:     extras.append(f"vol={d['volume']:,}")
            scanner_lines.append(f"  {sym:<6}  source={src}  {' '.join(extras)}")
        universe_table = (
            universe_table
            + "\n\nDYNAMIC SCANNER ADDITIONS today (these are NOT in the anchor list — "
            "they were added because of unusual flow/movement TODAY):\n"
            + "\n".join(scanner_lines)
        )
    n = len(snapshots)
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        now=now,
        session_label=session_label,
        equity=equity,
        open_positions=", ".join(open_positions) if open_positions else "(none)",
        n_open=len(open_positions),
        position_list=", ".join(open_positions) or "none",
        universe_table=universe_table,
        max_positions=max_positions,
        market_brief=market_brief.strip() or "(no brief available; vote on technicals only)",
        options_flow_block=options_flow_block.strip() or "(no options flow data this session)",
    )

    # Scale output token budget with universe size to avoid truncation.
    token_budget = _token_budget_for_universe_size(n)
    claude_payload, gpt_payload = await asyncio.gather(
        _call_claude(prompt, token_budget=token_budget),
        _call_gpt(prompt, token_budget=token_budget),
    )
    claude_votes = _vote_map(claude_payload)
    gpt_votes = _vote_map(gpt_payload)

    result: Dict[str, Dict] = {}
    for sym in snapshots.keys():
        c = claude_votes.get(sym, {"action": "HOLD", "confidence": 0, "reason": "no_vote"})
        g = gpt_votes.get(sym, {"action": "HOLD", "confidence": 0, "reason": "no_vote"})
        result[sym] = {"claude": c, "gpt": g, "consensus": consensus_for_symbol(c, g)}
    return result
