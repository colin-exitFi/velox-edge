"""Unusual Whales client — options flow context for Edge's contrarian voters.

Only the one method we actually need: pull a 4-hour summary of options flow
on a symbol (call/put premium ratio, retail vs institutional skew, dollar
volume) and format it as a one-line annotation. Inject into the consensus
prompt so Claude + GPT can distinguish:

  • A retail call buy-the-pump on yesterday's runner → strong fade signal
  • Institutional sweeps on the same name → don't fade, smart money's in

Cost: free (we have UW for ~1 month). Falls through silently if token absent
or rate-limited so Edge keeps trading on price alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from loguru import logger

from velox_edge import config


_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
_BASE = "https://api.unusualwhales.com/api"
_CACHE: Dict[str, Dict] = {}
_CACHE_TTL = 600  # 10 min — flow snapshots don't move that fast


@dataclass
class FlowSummary:
    symbol: str
    available: bool = False
    call_premium_usd: float = 0.0
    put_premium_usd: float = 0.0
    call_put_ratio: float = 0.0     # call premium / total. 1.0 = all calls, 0.5 = balanced
    n_alerts: int = 0
    largest_alert_usd: float = 0.0
    institutional_share: float = 0.0  # 0-1, share of premium with sweep/block flag
    error: str = ""

    def annotation(self) -> str:
        """One-line annotation for the consensus prompt."""
        if not self.available:
            return f"  {self.symbol:<6}  flow=N/A ({self.error or 'no signal'})"
        cp = "calls" if self.call_put_ratio >= 0.6 else ("puts" if self.call_put_ratio <= 0.4 else "balanced")
        ist = (
            "institutional sweeps" if self.institutional_share >= 0.5
            else "mostly retail" if self.institutional_share <= 0.2
            else "mixed"
        )
        return (
            f"  {self.symbol:<6}  ${(self.call_premium_usd + self.put_premium_usd)/1000:.0f}K total  "
            f"{cp} ({self.call_put_ratio*100:.0f}% calls)  {ist}  "
            f"{self.n_alerts} alerts  largest=${self.largest_alert_usd/1000:.0f}K"
        )


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {config.UW_API_TOKEN}",
        "Accept": "application/json",
    }


async def get_flow_summary(symbol: str, hours_back: int = 4) -> FlowSummary:
    """Pull recent flow alerts for symbol, summarize."""
    if not getattr(config, "UW_API_TOKEN", "") or not getattr(config, "UW_API_ENABLED", True):
        return FlowSummary(symbol=symbol, error="uw_disabled")

    cache_key = f"{symbol}_{hours_back}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        return cached["summary"]

    from datetime import datetime, timezone, timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/stock/{symbol}/flow-alerts",
                headers=_headers(),
                params={"limit": 50},
            )
            if r.status_code == 404:
                summary = FlowSummary(symbol=symbol, error="no_alerts")
                _CACHE[cache_key] = {"ts": time.time(), "summary": summary}
                return summary
            r.raise_for_status()
            payload = r.json() or {}
    except Exception as e:
        return FlowSummary(symbol=symbol, error=f"uw_error: {type(e).__name__}")

    alerts = payload.get("data") or []
    # Filter to last `hours_back` hours by ISO `created_at`
    recent = []
    for a in alerts:
        ts_str = a.get("created_at") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff_dt:
                recent.append(a)
        except Exception:
            continue

    if not recent:
        summary = FlowSummary(symbol=symbol, available=True, n_alerts=0)
        _CACHE[cache_key] = {"ts": time.time(), "summary": summary}
        return summary

    call_prem = 0.0
    put_prem = 0.0
    largest = 0.0
    institutional_prem = 0.0
    total_prem = 0.0

    for a in recent:
        try:
            prem = float(a.get("total_premium") or 0)
            ot = (a.get("type") or "").lower()
            rule = (a.get("alert_rule") or "").lower()
            has_sweep = bool(a.get("has_sweep"))
        except (TypeError, ValueError):
            continue
        total_prem += prem
        if ot == "call":
            call_prem += prem
        elif ot == "put":
            put_prem += prem
        if prem > largest:
            largest = prem
        # Institutional flags: sweep flag, block alert rules, repeated hits
        if has_sweep or any(kw in rule for kw in ("sweep", "block", "ascendingfill")):
            institutional_prem += prem

    cpr = (call_prem / total_prem) if total_prem else 0.5
    ist_share = (institutional_prem / total_prem) if total_prem else 0.0

    summary = FlowSummary(
        symbol=symbol,
        available=True,
        call_premium_usd=call_prem,
        put_premium_usd=put_prem,
        call_put_ratio=cpr,
        n_alerts=len(recent),
        largest_alert_usd=largest,
        institutional_share=ist_share,
    )
    _CACHE[cache_key] = {"ts": time.time(), "summary": summary}
    return summary


async def get_flow_block(symbols: List[str], hours_back: int = 4) -> str:
    """Pull flow for a batch of symbols sequentially with a small delay
    (UW rate-limits aggressive parallel calls). Format as a multi-line
    annotation for the Edge consensus prompt.
    """
    import asyncio
    if not symbols:
        return "(no symbols to query)"
    summaries = []
    for sym in symbols:
        try:
            s = await get_flow_summary(sym, hours_back=hours_back)
            summaries.append(s)
        except Exception as e:
            summaries.append(FlowSummary(symbol=sym, error=f"uw_error: {type(e).__name__}"))
        await asyncio.sleep(0.15)  # gentle rate limit
    lines = []
    avail = 0
    for s in summaries:
        if s.available:
            avail += 1
        lines.append(s.annotation())
    logger.info(f"📊 UW flow: pulled {avail}/{len(symbols)} symbols with data")
    return "\n".join(lines)
