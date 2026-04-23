"""Daily scanner — augments the anchor universe with today's most-active names.

Runs once at the start of each trading day. Pulls Alpaca's most-active screener
+ optional Polygon free tier for prev-day gainers/losers, filters for
tradeable liquidity, and returns 15-25 dynamic candidates with category tags.

These dynamic candidates get merged into the anchor universe for every session
that day — so the bot adapts to whatever the market is actually doing without
losing the consistency of the anchor list.

Cached for 24h (or until the next pre-market refresh).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

import httpx
from loguru import logger

from velox_edge import config
from velox_edge.universe import UNIVERSE


_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_CACHE: Dict[str, object] = {"ts": 0, "tickers": [], "details": []}
_CACHE_TTL = 6 * 3600  # 6 hours — covers a full session day


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }


async def _alpaca_most_actives(top: int = 50) -> List[Dict]:
    """Alpaca's most-actives screener (free with paper account).
    Returns list of {symbol, volume, trade_count}.
    """
    url = f"{config.ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, headers=_alpaca_headers(), params={"top": top, "by": "volume"})
            r.raise_for_status()
            data = r.json() or {}
        out = []
        for entry in data.get("most_actives") or []:
            sym = str(entry.get("symbol", "")).upper().strip()
            if not sym or "." in sym:  # skip BRK.B-style for now (works in anchor list only)
                continue
            out.append({
                "symbol": sym,
                "volume": int(entry.get("volume") or 0),
                "trade_count": int(entry.get("trade_count") or 0),
            })
        return out
    except Exception as e:
        logger.warning(f"Alpaca most-actives scan failed: {e}")
        return []


async def _alpaca_movers(top: int = 25) -> Dict[str, List[Dict]]:
    """Alpaca's market-movers (gainers + losers, free).
    Returns {gainers: [...], losers: [...]}.
    """
    url = f"{config.ALPACA_DATA_URL}/v1beta1/screener/stocks/movers"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, headers=_alpaca_headers(), params={"top": top})
            r.raise_for_status()
            data = r.json() or {}
        def _norm(rows):
            out = []
            for entry in rows or []:
                sym = str(entry.get("symbol", "")).upper().strip()
                if not sym or "." in sym:
                    continue
                out.append({
                    "symbol": sym,
                    "price": float(entry.get("price") or 0),
                    "change": float(entry.get("change") or 0),
                    "pct_change": float(entry.get("percent_change") or 0),
                })
            return out
        return {
            "gainers": _norm(data.get("gainers")),
            "losers": _norm(data.get("losers")),
        }
    except Exception as e:
        logger.warning(f"Alpaca movers scan failed: {e}")
        return {"gainers": [], "losers": []}


async def daily_scan(min_price: float = 5.0, target_count: int = 20) -> Dict[str, object]:
    """Returns {tickers: [list of new symbols not in anchor], details: [...]}."""
    if (time.time() - _CACHE["ts"]) < _CACHE_TTL and _CACHE["tickers"]:
        return {"tickers": list(_CACHE["tickers"]), "details": list(_CACHE["details"])}

    actives = await _alpaca_most_actives(top=80)
    movers = await _alpaca_movers(top=25)

    seen: Set[str] = set(UNIVERSE)
    out_tickers: List[str] = []
    out_details: List[Dict] = []

    def _add(sym: str, source: str, extra: Optional[Dict] = None):
        if sym in seen:
            return
        seen.add(sym)
        out_tickers.append(sym)
        out_details.append({"symbol": sym, "source": source, **(extra or {})})

    # Gainers and losers first — these are the names where active management has a real shot
    for g in movers.get("gainers", []):
        if g["price"] < min_price:
            continue
        _add(g["symbol"], "gainer", {"price": g["price"], "pct_change": g["pct_change"]})

    for l in movers.get("losers", []):
        if l["price"] < min_price:
            continue
        _add(l["symbol"], "loser", {"price": l["price"], "pct_change": l["pct_change"]})

    # Most-actives fill the rest of the slots
    for a in actives:
        if len(out_tickers) >= target_count:
            break
        _add(a["symbol"], "most_active", {"volume": a["volume"]})

    out_tickers = out_tickers[:target_count]
    out_details = out_details[:target_count]

    _CACHE["ts"] = time.time()
    _CACHE["tickers"] = out_tickers
    _CACHE["details"] = out_details
    logger.info(
        f"🔭 Daily scan: {len(out_tickers)} new tickers added to anchor "
        f"({sum(1 for d in out_details if d['source']=='gainer')} gainers, "
        f"{sum(1 for d in out_details if d['source']=='loser')} losers, "
        f"{sum(1 for d in out_details if d['source']=='most_active')} most-active)"
    )
    return {"tickers": out_tickers, "details": out_details}


def cached_scan() -> Dict[str, object]:
    return {"tickers": list(_CACHE["tickers"]), "details": list(_CACHE["details"])}
