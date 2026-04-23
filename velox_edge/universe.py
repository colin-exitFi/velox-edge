"""Velox Edge anchor universe — deliberately tiny.

Edge's mandate is contrarian: fade the crowd, hunt extremes. That work
happens on names that are MOVING TODAY, not on a curated watchlist. So the
anchor here is just the broad-market ETFs (for context + hedging). Everything
else comes from the daily scanner.

The scanner is what makes Edge edgy.
"""

ETFS = [
    "SPY", "QQQ", "IWM", "VIXY",  # VIXY for vol context
]

UNIVERSE = list(ETFS)

CATEGORY_OF = {
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "VIXY": "vol_proxy",
}

# Compatibility shims (consensus.py + sizing.py reference these)
SP100 = []
MID_CAP_AI_SOFTWARE = []
MID_CAP_SEMIS = []
MID_CAP_FINTECH_NEWAGE = []
MID_CAP_HIGH_BETA = []
MID_CAP_BIOTECH = []
MID_CAP_CONSUMER_TRAVEL = []
DEFENSIVE = []
MEGA_CAP_TECH = []
AI_NARRATIVE_MIDCAP = []
HIGH_BETA_NARRATIVE = []
