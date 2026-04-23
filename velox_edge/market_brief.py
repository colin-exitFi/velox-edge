"""Perplexity market-brief layer.

Runs once per consensus session. Asks Perplexity (sonar-pro) for the most
material short-term catalysts on the 40-ticker universe and the macro tape.
The returned brief is injected into both Claude's and GPT's prompts so they
vote with context instead of in a vacuum.

Cost discipline:
 - 1 call per session × 5 sessions/day = 5 calls/day
 - ~1000 input + 800 output tokens per call
 - sonar-pro: $3/M input + $15/M output + $0.006/request
 - ~$4/month at our cadence
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import httpx
from loguru import logger

from velox_edge import config


_TIMEOUT = httpx.Timeout(30.0, connect=8.0)


@dataclass
class MarketBrief:
    text: str
    citations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    session_label: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "timestamp": self.timestamp,
            "session_label": self.session_label,
            "error": self.error,
        }


def _empty(reason: str = "") -> MarketBrief:
    return MarketBrief(
        text="(no market brief available — voting on technicals only)",
        error=reason,
    )


PROMPT = """You are the market-context analyst for Velox Edge, an autonomous paper trading bot.
The bot trades a fixed universe of 40 large/mid-cap US tickers at 5 sessions per trading day.
This call runs ONCE per session, RIGHT BEFORE two AI voters (Claude + GPT) score each ticker.
Your job is to give them the live context they would otherwise miss.

CURRENT TIME: {now} ET ({session_label} session)
UNIVERSE: {universe_str}

Produce a tight, structured brief covering ONLY what materially affects short-term (30-90 minute)
equity decisions in the next 4 hours. Skip anything generic. No platitudes.

Cover these in order, each in 1-2 sentences:
  1. **Macro tape right now** — what indices are doing, regime tone (risk-on / risk-off / mixed),
     any pending Fed / data prints in the next 4 hours.
  2. **Material catalysts on universe tickers** — earnings dropping today, analyst actions,
     guidance, M&A, FDA, congressional/insider, anything that should move price intraday.
     ONLY list tickers from our universe; ignore the rest of the market.
  3. **Sector themes in motion** — which sectors are running or breaking, with the actual reason
     (e.g. "AI capex narrative on hyperscaler earnings beats" not "tech is up").
  4. **Risk flags** — anything that says "be smaller / wait" today
     (Fed minutes pending, low volume holiday, geopolitical, etc.).

Format: short markdown bullets, no preamble, no closing summary. Maximum 250 words.
If a section has nothing material, write "Nothing material." for that section. Do not pad."""


async def get_market_brief(
    universe: List[str], session_label: str
) -> MarketBrief:
    if not config.PERPLEXITY_API_KEY or not config.MARKET_BRIEF_ENABLED:
        return _empty("perplexity_disabled")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = PROMPT.format(
        now=now,
        session_label=session_label,
        universe_str=", ".join(universe),
    )

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.PERPLEXITY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            payload = r.json()
            text = payload["choices"][0]["message"]["content"].strip()
            citations = payload.get("citations") or []
            logger.info(
                f"📰 Market brief from {config.PERPLEXITY_MODEL} in "
                f"{time.time()-started:.1f}s ({len(text)} chars, {len(citations)} citations)"
            )
            return MarketBrief(
                text=text,
                citations=[str(c) for c in citations][:10],
                session_label=session_label,
            )
    except Exception as e:
        logger.error(f"Market brief failed: {e}")
        return _empty(f"perplexity_error: {e}")
