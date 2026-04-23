"""Position sizing + concentration guard.

Two ideas inherited from velox-classic, deliberately rebuilt small:
  1. Conviction-based sizing — higher consensus confidence = bigger position.
  2. Sector concentration guard — cap exposure to any one universe category
     so the bot can't go 32% AI mid-cap on a single bullish session.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from velox_edge import config
from velox_edge.universe import CATEGORY_OF


def conviction_size_pct(consensus_confidence: float) -> float:
    """Linear scale from MIN_PCT (at MIN_CONFIDENCE) to MAX_PCT (at 100%).

    Examples (defaults: min 3%, max 6%, threshold 60%):
      60% conf → 3.00%
      70% conf → 3.75%
      80% conf → 4.50%
      90% conf → 5.25%
     100% conf → 6.00%
    """
    floor = config.MIN_CONSENSUS_CONFIDENCE
    if consensus_confidence < floor:
        return 0.0
    span = max(1.0, 100.0 - floor)
    t = (consensus_confidence - floor) / span
    t = max(0.0, min(1.0, t))
    return config.POSITION_SIZE_MIN_PCT + t * (
        config.POSITION_SIZE_MAX_PCT - config.POSITION_SIZE_MIN_PCT
    )


def size_position(
    equity: float, price: float, consensus_confidence: float
) -> Tuple[float, float]:
    """Return (qty, intended_pct) for a new entry. qty=0 if too small."""
    if price <= 0 or equity <= 0:
        return 0.0, 0.0
    pct = conviction_size_pct(consensus_confidence)
    if pct <= 0:
        return 0.0, 0.0
    notional = equity * (pct / 100.0)
    qty = max(0, int(notional // price))
    if qty * price < 50:  # ignore microscopic positions
        return 0.0, pct
    return float(qty), pct


def category_exposure(positions: List[Dict]) -> Dict[str, float]:
    """Sum market_value per universe category from current broker positions."""
    out: Dict[str, float] = {}
    for p in positions or []:
        sym = p.get("symbol", "")
        cat = CATEGORY_OF.get(sym, "other")
        try:
            mv = abs(float(p.get("market_value") or 0))
        except (TypeError, ValueError):
            mv = 0.0
        out[cat] = out.get(cat, 0.0) + mv
    return out


def concentration_block_reason(
    symbol: str,
    intended_notional: float,
    equity: float,
    positions: List[Dict],
) -> Optional[str]:
    """Return a reason string if entering this position would breach the cap,
    else None.

    Cap is MAX_CATEGORY_EXPOSURE_PCT of equity per category (default 35%).
    """
    if equity <= 0:
        return None
    cap_pct = config.MAX_CATEGORY_EXPOSURE_PCT
    cap_dollars = equity * (cap_pct / 100.0)
    cat = CATEGORY_OF.get(symbol, "other")
    current = category_exposure(positions).get(cat, 0.0)
    projected = current + intended_notional
    if projected > cap_dollars:
        return (
            f"concentration_cap_{cat}: would push {cat} to "
            f"${projected:,.0f} (cap ${cap_dollars:,.0f} / {cap_pct:.0f}% of equity)"
        )
    return None
