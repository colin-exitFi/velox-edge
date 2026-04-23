"""Runtime configuration loaded from .env."""

import os
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=True)


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


# ── Alpaca ─────────────────────────────────────────────────────────
ALPACA_API_KEY = _str("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _str("ALPACA_SECRET_KEY")
ALPACA_PAPER = _bool("ALPACA_PAPER", True)
ALPACA_BASE_URL = (
    "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
)
ALPACA_DATA_URL = "https://data.alpaca.markets"

# ── AI providers ───────────────────────────────────────────────────
ANTHROPIC_API_KEY = _str("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _str("OPENAI_API_KEY")
PERPLEXITY_API_KEY = _str("PERPLEXITY_API_KEY")
# Defaults updated 2026-04-22 to current top-tier models.
# Claude Opus 4.7 + GPT-5.4 are the "voters" in the consensus.
# Perplexity sonar-pro is the "context layer" — runs once per session, injects
# a real-time market brief into both voters' prompts.
ANTHROPIC_MODEL = _str("ANTHROPIC_MODEL", "claude-opus-4-7")
OPENAI_MODEL = _str("OPENAI_MODEL", "gpt-5.4")
PERPLEXITY_MODEL = _str("PERPLEXITY_MODEL", "sonar-pro")
MARKET_BRIEF_ENABLED = _bool("MARKET_BRIEF_ENABLED", bool(PERPLEXITY_API_KEY))

# ── Trading (CONTRARIAN profile — wider stops, bigger size, longer hold) ──
POSITION_SIZE_PCT = _float("POSITION_SIZE_PCT", 7.0)
MAX_CONCURRENT_POSITIONS = _int("MAX_CONCURRENT_POSITIONS", 6)
# Higher confidence bar than vanilla — we fade extremes, only on conviction
MIN_CONSENSUS_CONFIDENCE = _float("MIN_CONSENSUS_CONFIDENCE", 65)
PAPER_STARTING_EQUITY = _float("PAPER_STARTING_EQUITY", 25000)

# Conviction sizing — Edge swings bigger on the highest-conviction trades.
# 65% conf → 5%; 100% conf → 10%
POSITION_SIZE_MIN_PCT = _float("POSITION_SIZE_MIN_PCT", 5.0)
POSITION_SIZE_MAX_PCT = _float("POSITION_SIZE_MAX_PCT", 10.0)

# Concentration: more permissive since Edge's universe is small/dynamic.
MAX_CATEGORY_EXPOSURE_PCT = _float("MAX_CATEGORY_EXPOSURE_PCT", 50.0)

# Hold profile — Edge can carry winners overnight up to N days for full
# mean-reversion. Force-flatten only on day N if not already exited.
EDGE_MAX_HOLD_DAYS = _int("EDGE_MAX_HOLD_DAYS", 3)
EDGE_FORCE_FLATTEN_AT_EOD = _bool("EDGE_FORCE_FLATTEN_AT_EOD", False)

# Daily review: same as core
DAILY_REVIEW_ENABLED = _bool("DAILY_REVIEW_ENABLED", bool(ANTHROPIC_API_KEY))
DAILY_REVIEW_HOUR_ET = _int("DAILY_REVIEW_HOUR_ET", 16)
DAILY_REVIEW_MIN_ET = _int("DAILY_REVIEW_MIN_ET", 5)

# ── Ratchet (WIDER for contrarian fades — squeeze risk is real) ────
RATCHET_HARD_STOP_PCT = _float("RATCHET_HARD_STOP_PCT", -3.00)
RATCHET_ACTIVATION_PCT = _float("RATCHET_ACTIVATION_PCT", 1.00)
RATCHET_TRAIL_PCT = _float("RATCHET_TRAIL_PCT", 2.00)
RATCHET_INITIAL_FLOOR_PCT = _float("RATCHET_INITIAL_FLOOR_PCT", 0.25)
RATCHET_MIN_HOLD_SECONDS = _int("RATCHET_MIN_HOLD_SECONDS", 300)

# ── Dashboard ──────────────────────────────────────────────────────
DASHBOARD_HOST = _str("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = _int("DASHBOARD_PORT", 8423)
DASHBOARD_TOKEN = _str("DASHBOARD_TOKEN", "")

# ── Operational ────────────────────────────────────────────────────
TRADING_HALTED = _bool("TRADING_HALTED", False)
LOG_LEVEL = _str("LOG_LEVEL", "INFO")

DATA_DIR = _root / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "velox_edge.db"
EQUITY_HISTORY_PATH = DATA_DIR / "equity_history.json"
