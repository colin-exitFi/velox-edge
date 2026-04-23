"""Persistent state: SQLite for trades/decisions/skips, JSON for equity/positions.

Two principles:
 1. Every decision the bot makes is recorded. Every trade is attributable to
    a specific consensus event. Skips (where models disagreed) are equally
    important data — without them you can't measure whether the consensus
    filter is adding alpha.
 2. The schema is small. Five tables, no migrations, no ORM. If something is
    confusing in here, the schema is wrong.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger

from velox_edge import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    label TEXT,
    equity_at_start REAL,
    equity_at_end REAL,
    trades_opened INTEGER DEFAULT 0,
    trades_skipped INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    claude_action TEXT,
    claude_confidence REAL,
    claude_reason TEXT,
    gpt_action TEXT,
    gpt_confidence REAL,
    gpt_reason TEXT,
    consensus_action TEXT,        -- BUY / SHORT / HOLD / SKIP
    consensus_confidence REAL,
    skip_reason TEXT,             -- why we did NOT trade (disagreement / low conf / cap reached)
    executed INTEGER DEFAULT 0,   -- 1 if a trade was actually placed
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,           -- 'long' or 'short'
    entry_price REAL NOT NULL,
    entry_time REAL NOT NULL,
    qty REAL NOT NULL,
    exit_price REAL,
    exit_time REAL,
    exit_reason TEXT,
    pnl REAL,
    pnl_pct REAL,
    hold_seconds REAL,
    consensus_confidence REAL,
    claude_vote TEXT,
    gpt_vote TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    level TEXT NOT NULL,          -- info / warn / error
    event TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS market_briefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp REAL NOT NULL,
    session_label TEXT,
    text TEXT NOT NULL,
    citations TEXT,               -- JSON array of citation URLs
    error TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_briefs_timestamp ON market_briefs(timestamp);

CREATE TABLE IF NOT EXISTS daily_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    review_date TEXT NOT NULL,    -- 'YYYY-MM-DD' for easy lookup
    text TEXT NOT NULL,
    day_pnl REAL,
    n_closed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_reviews_date ON daily_reviews(review_date);

CREATE TABLE IF NOT EXISTS game_film (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    insights_json TEXT NOT NULL    -- full insights dict serialized
);

CREATE INDEX IF NOT EXISTS idx_game_film_ts ON game_film(timestamp);
"""


@contextmanager
def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)
    logger.info(f"State DB initialized at {config.DB_PATH}")


# ── Sessions ───────────────────────────────────────────────────────


def start_session(label: str, equity: float) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions (started_at, label, equity_at_start) VALUES (?, ?, ?)",
            (time.time(), label, equity),
        )
        return cur.lastrowid


def end_session(session_id: int, equity: float, opened: int, skipped: int):
    with _conn() as c:
        c.execute(
            """UPDATE sessions
               SET ended_at = ?, equity_at_end = ?, trades_opened = ?, trades_skipped = ?
               WHERE id = ?""",
            (time.time(), equity, opened, skipped, session_id),
        )


# ── Decisions ──────────────────────────────────────────────────────


def record_decision(
    session_id: int,
    symbol: str,
    price: float,
    claude_vote: Dict,
    gpt_vote: Dict,
    consensus_action: str,
    consensus_confidence: float,
    skip_reason: str = "",
    executed: bool = False,
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO decisions
               (session_id, timestamp, symbol, price,
                claude_action, claude_confidence, claude_reason,
                gpt_action, gpt_confidence, gpt_reason,
                consensus_action, consensus_confidence, skip_reason, executed)
               VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?)""",
            (
                session_id, time.time(), symbol, price,
                claude_vote.get("action", "HOLD"),
                float(claude_vote.get("confidence", 0)),
                str(claude_vote.get("reason", ""))[:500],
                gpt_vote.get("action", "HOLD"),
                float(gpt_vote.get("confidence", 0)),
                str(gpt_vote.get("reason", ""))[:500],
                consensus_action,
                consensus_confidence,
                skip_reason,
                1 if executed else 0,
            ),
        )
        return cur.lastrowid


def recent_decisions(limit: int = 50) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def recent_skips(limit: int = 100) -> List[Dict]:
    """Skips are decisions where we did NOT trade — the consensus filter's work."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM decisions WHERE executed = 0 AND skip_reason != '' "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Trades ─────────────────────────────────────────────────────────


def record_trade_open(
    decision_id: Optional[int],
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    consensus_confidence: float,
    claude_vote: Dict,
    gpt_vote: Dict,
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO trades
               (decision_id, symbol, side, entry_price, entry_time, qty,
                consensus_confidence, claude_vote, gpt_vote)
               VALUES (?,?,?,?,?,?, ?,?,?)""",
            (
                decision_id, symbol, side, entry_price, time.time(), qty,
                consensus_confidence,
                f"{claude_vote.get('action','?')}@{claude_vote.get('confidence',0):.0f}",
                f"{gpt_vote.get('action','?')}@{gpt_vote.get('confidence',0):.0f}",
            ),
        )
        return cur.lastrowid


def record_trade_close(trade_id: int, exit_price: float, exit_reason: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            return
        entry_price = float(row["entry_price"])
        qty = float(row["qty"])
        side = row["side"]
        if side == "long":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price else 0
        else:
            pnl = (entry_price - exit_price) * qty
            pnl_pct = (1 - exit_price / entry_price) * 100 if entry_price else 0
        c.execute(
            """UPDATE trades
               SET exit_price = ?, exit_time = ?, exit_reason = ?,
                   pnl = ?, pnl_pct = ?, hold_seconds = ?
               WHERE id = ?""",
            (
                exit_price, time.time(), exit_reason,
                pnl, pnl_pct, time.time() - float(row["entry_time"]),
                trade_id,
            ),
        )


def get_open_trades() -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def find_open_trade(symbol: str) -> Optional[Dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM trades WHERE symbol = ? AND exit_time IS NULL "
            "ORDER BY entry_time DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None


def recent_closed_trades(limit: int = 50) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL "
            "ORDER BY exit_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def trade_summary() -> Dict:
    with _conn() as c:
        rows = c.execute(
            "SELECT pnl, pnl_pct, claude_vote, gpt_vote FROM trades WHERE exit_time IS NOT NULL"
        ).fetchall()
    if not rows:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
        }
    pnls = [float(r["pnl"] or 0) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(pnls) * 100) if pnls else 0,
        "total_pnl": sum(pnls),
        "avg_win": (sum(wins) / len(wins)) if wins else 0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0,
    }


def model_scoreboard() -> Dict:
    """Compare Claude-solo vs GPT-solo vs Consensus on closed trades.

    Each closed trade is attributed to whichever model (or both) voted in the
    direction that the trade ended up taking. We measure the realized P&L
    contribution per model.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT pnl, claude_vote, gpt_vote, side FROM trades WHERE exit_time IS NOT NULL"
        ).fetchall()
    claude_pnls, gpt_pnls, agreed_pnls = [], [], []
    for r in rows:
        pnl = float(r["pnl"] or 0)
        side_dir = "BUY" if r["side"] == "long" else "SHORT"
        cv = (r["claude_vote"] or "").split("@")[0]
        gv = (r["gpt_vote"] or "").split("@")[0]
        if cv == side_dir:
            claude_pnls.append(pnl)
        if gv == side_dir:
            gpt_pnls.append(pnl)
        if cv == side_dir and gv == side_dir:
            agreed_pnls.append(pnl)

    def _stats(arr):
        if not arr:
            return {"trades": 0, "wins": 0, "win_rate": 0, "total_pnl": 0}
        wins = [p for p in arr if p > 0]
        return {
            "trades": len(arr),
            "wins": len(wins),
            "win_rate": len(wins) / len(arr) * 100,
            "total_pnl": sum(arr),
        }

    return {
        "claude": _stats(claude_pnls),
        "gpt": _stats(gpt_pnls),
        "consensus": _stats(agreed_pnls),
    }


# ── Audit log (operational events) ─────────────────────────────────


def audit(event: str, level: str = "info", detail: str = ""):
    with _conn() as c:
        c.execute(
            "INSERT INTO audit (timestamp, level, event, detail) VALUES (?,?,?,?)",
            (time.time(), level, event, detail[:1000]),
        )


def recent_audit(limit: int = 50) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM audit ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Market briefs ──────────────────────────────────────────────────


def record_market_brief(
    session_id: Optional[int],
    session_label: str,
    text: str,
    citations: Optional[List[str]] = None,
    error: str = "",
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO market_briefs
               (session_id, timestamp, session_label, text, citations, error)
               VALUES (?,?,?,?,?,?)""",
            (
                session_id, time.time(), session_label, text,
                json.dumps(citations or []), error,
            ),
        )
        return cur.lastrowid


def latest_market_brief() -> Optional[Dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM market_briefs ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["citations"] = json.loads(d.get("citations") or "[]")
        except Exception:
            d["citations"] = []
        return d


def recent_market_briefs(limit: int = 10) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM market_briefs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["citations"] = json.loads(d.get("citations") or "[]")
        except Exception:
            d["citations"] = []
        out.append(d)
    return out


# ── Daily reviews ──────────────────────────────────────────────────


def record_daily_review(text: str, day_pnl: float, n_closed: int) -> int:
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y-%m-%d")
    with _conn() as c:
        # Replace today's review if one already exists (idempotent re-runs)
        c.execute("DELETE FROM daily_reviews WHERE review_date = ?", (today,))
        cur = c.execute(
            """INSERT INTO daily_reviews (timestamp, review_date, text, day_pnl, n_closed)
               VALUES (?,?,?,?,?)""",
            (time.time(), today, text, day_pnl, n_closed),
        )
        return cur.lastrowid


def latest_daily_review() -> Optional[Dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM daily_reviews ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def recent_daily_reviews(limit: int = 14) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM daily_reviews ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Game film ──────────────────────────────────────────────────────


def record_game_film(insights: Dict) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO game_film (timestamp, insights_json) VALUES (?, ?)",
            (time.time(), json.dumps(insights)),
        )
        return cur.lastrowid


def latest_game_film() -> Optional[Dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM game_film ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["insights"] = json.loads(d.get("insights_json") or "{}")
        except Exception:
            d["insights"] = {}
        d.pop("insights_json", None)
        return d


# ── Equity history (JSON for fast charting) ────────────────────────


def append_equity_point(equity: float, spy_price: float = 0.0):
    path = config.EQUITY_HISTORY_PATH
    history: List[Dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except Exception:
            history = []
    history.append({"timestamp": time.time(), "equity": equity, "spy": spy_price})
    # keep last 30 days of points (assume 1 sample per session, ~5/day = ~150 points)
    history = history[-2000:]
    path.write_text(json.dumps(history))


def equity_history() -> List[Dict]:
    path = config.EQUITY_HISTORY_PATH
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []
