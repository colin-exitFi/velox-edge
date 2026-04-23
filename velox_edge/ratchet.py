"""Profit ratchet — the proven exit logic from velox-classic.

Behavior:
 - Hard stop at -0.75% of entry
 - At +0.30% peak, ratchet activates and starts trailing
 - Trail at -1.00% from peak (giving back at most 1% from MFE)
 - After activation, never let stop fall below +0.10% from entry (locked profit floor)
 - Min hold of 120s — don't churn out within 2 minutes of entry

Pure stateless function. Caller owns the position state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from velox_edge import config


@dataclass
class RatchetState:
    entry_price: float
    entry_time: float
    side: str                 # 'long' or 'short'
    peak_pnl_pct: float = 0.0
    activated: bool = False


def pnl_pct(state: RatchetState, current_price: float) -> float:
    if state.entry_price <= 0:
        return 0.0
    if state.side == "long":
        return (current_price / state.entry_price - 1) * 100
    return (1 - current_price / state.entry_price) * 100


def update_peak(state: RatchetState, current_price: float) -> RatchetState:
    p = pnl_pct(state, current_price)
    if p > state.peak_pnl_pct:
        state.peak_pnl_pct = p
    if not state.activated and state.peak_pnl_pct >= config.RATCHET_ACTIVATION_PCT:
        state.activated = True
    return state


def should_exit(state: RatchetState, current_price: float) -> Optional[str]:
    """Return exit reason string if we should close, else None."""
    held = time.time() - state.entry_time
    p = pnl_pct(state, current_price)

    # Hard stop fires regardless of min hold — capital protection wins
    if p <= config.RATCHET_HARD_STOP_PCT:
        return "hard_stop"

    if held < config.RATCHET_MIN_HOLD_SECONDS:
        return None

    if state.activated:
        # Trailing exit: peak minus trail width, but never below the locked floor
        trail_floor = state.peak_pnl_pct - config.RATCHET_TRAIL_PCT
        locked_floor = config.RATCHET_INITIAL_FLOOR_PCT
        effective_floor = max(trail_floor, locked_floor)
        if p <= effective_floor:
            return "trailing_stop"

    return None
