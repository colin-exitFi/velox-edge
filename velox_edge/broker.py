"""Alpaca broker — quotes, snapshots, account, orders.

Thin wrapper. We use raw HTTP because the alpaca-py SDK pulls in heavy
deps and we want this codebase to fit in your head.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import httpx
from loguru import logger

from velox_edge import config


_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


# ── Account ────────────────────────────────────────────────────────


async def get_account() -> Dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{config.ALPACA_BASE_URL}/v2/account", headers=_headers())
        r.raise_for_status()
        return r.json()


async def get_equity() -> float:
    try:
        acct = await get_account()
        return float(acct.get("equity", 0))
    except Exception as e:
        logger.error(f"get_equity failed: {e}")
        return 0.0


# ── Market data ────────────────────────────────────────────────────


async def get_snapshots(symbols: List[str]) -> Dict[str, Dict]:
    """Return snapshot per symbol with: price, change_pct, volume, prev_close, vwap.

    Uses Alpaca's IEX feed (free). One batched call.
    """
    if not symbols:
        return {}
    url = f"{config.ALPACA_DATA_URL}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols), "feed": "iex"}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url, headers=_headers(), params=params)
        r.raise_for_status()
        raw = r.json() or {}

    out: Dict[str, Dict] = {}
    for sym, snap in raw.items():
        if not snap:
            continue
        latest = snap.get("latestTrade") or {}
        prev = snap.get("prevDailyBar") or {}
        day = snap.get("dailyBar") or {}
        price = float(latest.get("p") or day.get("c") or 0.0)
        prev_close = float(prev.get("c") or 0.0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        out[sym] = {
            "symbol": sym,
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "volume": int(day.get("v") or 0),
            "prev_volume": int(prev.get("v") or 0),
            "vwap": float(day.get("vw") or 0.0),
            "high": float(day.get("h") or 0.0),
            "low": float(day.get("l") or 0.0),
        }
    return out


async def get_price(symbol: str) -> float:
    snaps = await get_snapshots([symbol])
    return float((snaps.get(symbol) or {}).get("price", 0.0))


# ── Positions ──────────────────────────────────────────────────────


async def get_positions() -> List[Dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{config.ALPACA_BASE_URL}/v2/positions", headers=_headers())
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json() or []


async def get_position(symbol: str) -> Optional[Dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(
            f"{config.ALPACA_BASE_URL}/v2/positions/{symbol}", headers=_headers()
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


# ── Orders ─────────────────────────────────────────────────────────


async def submit_market_order(
    symbol: str, qty: float, side: str, time_in_force: str = "day"
) -> Optional[Dict]:
    """Submit a market order. side='buy' or 'sell'. Returns the order dict or None."""
    body = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": time_in_force,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.post(
                f"{config.ALPACA_BASE_URL}/v2/orders", headers=_headers(), json=body
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Order rejected for {symbol} {side} {qty}: "
                f"{e.response.status_code} {e.response.text[:200]}"
            )
            return None
        except Exception as e:
            logger.error(f"Order error for {symbol}: {e}")
            return None


async def close_position(symbol: str) -> Optional[Dict]:
    """Close a position via Alpaca's DELETE /v2/positions/{symbol}."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.delete(
                f"{config.ALPACA_BASE_URL}/v2/positions/{symbol}", headers=_headers()
            )
            if r.status_code in (200, 207):
                return r.json()
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Close position error for {symbol}: {e}")
            return None


# ── Clock / market hours ───────────────────────────────────────────


async def get_clock() -> Dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{config.ALPACA_BASE_URL}/v2/clock", headers=_headers())
        r.raise_for_status()
        return r.json()


async def is_market_open() -> bool:
    try:
        c = await get_clock()
        return bool(c.get("is_open"))
    except Exception as e:
        logger.error(f"is_market_open failed: {e}")
        return False
