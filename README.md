# Velox Edge

A consensus paper-trading bot. Two AI models (Claude + GPT) vote on every trade. The bot only enters when they agree at high confidence. The profit ratchet manages every exit.

**One job:** beat SPY on a paper account over 30 days, on a fixed 40-ticker universe, using consensus from two AIs and a profit ratchet inherited from velox-classic.

If it doesn't beat SPY in 30 days, it gets killed and replaced. The bar doesn't negotiate.

## Architecture

5 sessions per trading day at **9:35, 11:00, 13:30, 15:00, 15:45 ET**. At each session:

1. Pull a snapshot for all 40 tickers from Alpaca (free IEX feed).
2. Send the entire universe to Claude *and* GPT in one prompt each.
3. Each model returns a vote per ticker: `BUY` / `SHORT` / `HOLD` / `EXIT` with confidence 0-100.
4. **Consensus = both models agree on action AND mean confidence ≥ 60.**
5. Place market orders for new entries; sized at 4% of equity, capped at 8 concurrent positions.
6. Between sessions, ratchet ticks every 30s on open positions (-0.75% hard stop, +0.30% activation, 1% trail).
7. Force-flatten everything at 15:55 ET (no overnight in v1).

## What's in the box

```
velox_core/
├── config.py        — env-driven settings
├── universe.py      — the 40-ticker list (3 ETFs + 10 mega-cap tech + 10 AI-mid + 10 high-beta + 7 defensive)
├── broker.py        — Alpaca quotes / orders (httpx, no SDK)
├── consensus.py     — Claude + GPT batch scoring + voting logic
├── ratchet.py       — proven exit values from velox-classic 397-trade dataset
├── state.py         — SQLite for trades/decisions/skips, JSON for equity history
├── main.py          — session scheduler + ratchet tick loop
└── dashboard.py     — single-page UI (FastAPI + vanilla JS + Chart.js)

run.py               — launcher (bot + dashboard in one process)
deploy/velox-core.service  — systemd unit
```

That's it. ~1,200 lines of Python total. You can read all of it.

## What it deliberately does NOT have

- No state machine, no setup funnel, no SQLite of 84k events, no book allocator
- No mode classifier, no pre-trade cost gate (Almgren-Chriss), no concentration guard
- No 8 desks, no multi-tier risk tiers
- No X / UW / Polygon / Perplexity / Grok feeds
- No options engine, no extended hours trading, no overnight positions
- No scanner — the universe is fixed at 40 tickers

If any of the above features prove necessary, they earn their re-add through evidence on a future bot.

## Cost target

- Alpaca paper: $0
- Anthropic (Claude Sonnet): ~$30-50/mo (1 batch call × 5 sessions/day)
- OpenAI (GPT-5.4-mini): ~$5-15/mo (same)
- VPS: $12/mo
- **Total: under $80/mo, target $50/mo**

## The honest bar

Velox Edge lives or dies on a single comparison: **its 30-day equity curve vs SPY's 30-day equity curve over the same window.**

ARC's 30-day post-mortem ([arc.donothinglabs.com/postmortem](https://arc.donothinglabs.com/postmortem)) showed both Anthropic (+4.45%) and OpenAI (+2.48%) councils losing to SPY (+8.36%) over the same period. The default outcome is losing to the index. That's the bar we're trying to clear.

If at day 30 Velox Edge has not beaten SPY, it gets killed. No tuning marathon. No "we just need to add this one feed." Kill, learn, fork, try the next architecture.

## Setup

```bash
git clone https://github.com/colin-exitFi/velox-core.git
cd velox-core
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env with your Alpaca paper + Anthropic + OpenAI keys
.venv/bin/python run.py
```

Dashboard at `http://localhost:8422?token=<your-token>`.

## VPS deploy

```bash
sudo cp deploy/velox-core.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable velox-core
sudo systemctl start velox-core
sudo journalctl -u velox-core -f
```

## Kill switch

Set `TRADING_HALTED=true` in `.env` and restart, OR click the red **Halt trading** button on the dashboard. Existing positions stay protected by the ratchet; new entries are blocked.
