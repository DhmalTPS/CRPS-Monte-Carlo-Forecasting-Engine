"""Static dictionaries (permitted artifact class: "dictionaries and tables"; no neural weights).

Hand-written from central-bank communication conventions; no fitting to any unit's outcome.
"""

from __future__ import annotations

# asset -> vocabulary that plausibly concerns it
ASSET_TERMS: dict[str, tuple[str, ...]] = {
    "UST": ("treasury", "treasuries", "yield", "yields", "federal funds", "fomc", "policy rate",
            "rate hike", "rate cut", "tightening", "easing", "balance sheet", "term premium",
            "inflation", "curve", "duration", "bond"),
    "EUR": ("euro", "ecb", "eurosystem", "euro area", "eurozone", "governing council", "lagarde",
            "draghi", "trichet", "bund"),
    "GBP": ("sterling", "pound", "bank of england", "mpc", "gilt", "united kingdom", "uk "),
    "JPY": ("yen", "bank of japan", "boj", "kuroda", "ueda", "japan", "yield curve control",
            "jgb"),
    "CHF": ("franc", "swiss national bank", "snb", "switzerland", "minimum exchange rate"),
    "AUD": ("australian dollar", "reserve bank of australia", "rba", "australia", "iron ore",
            "commodity"),
    "CAD": ("canadian dollar", "bank of canada", "canada", "oil", "crude"),
    "NZD": ("new zealand", "rbnz", "kiwi"),
    "SEK": ("krona", "riksbank", "sweden", "swedish"),
    "NOK": ("krone", "norges bank", "norway", "norwegian", "oil", "crude"),
    "DKK": ("danish", "denmark", "danmarks nationalbank", "peg"),
    "CNY": ("renminbi", "yuan", "rmb", "people's bank of china", "pboc", "china", "chinese",
            "fixing", "central parity"),
    "BRL": ("brazil", "real", "banco central do brasil", "emerging market"),
    "INR": ("india", "rupee", "reserve bank of india", "emerging market"),
    "MKT": ("equity", "equities", "stock", "stocks", "s&p", "risk assets", "market volatility",
            "earnings", "sell-off", "selloff"),
    "HML": ("value", "growth stocks", "rotation", "discount rate"),
    "MOM": ("momentum", "rotation", "reversal", "crowded"),
    "SMB": ("small cap", "small-cap", "size"),
    "QMJ": ("quality", "flight to quality"),
    "BAB": ("low volatility", "leverage", "beta"),
    "CPI": ("inflation", "consumer price", "cpi", "prices", "energy prices", "shelter"),
    "PCE": ("inflation", "pce", "personal consumption"),
    "NFP": ("payroll", "payrolls", "employment", "jobs", "labor market", "hiring", "layoffs"),
    "UNRATE": ("unemployment", "jobless", "labor market", "employment"),
}

HAWKISH = ("tighten", "tightening", "raise the target", "increase in the target", "rate increase",
           "rate increases", "hike", "hikes", "hiking", "inflation pressures", "upside risks to inflation",
           "elevated inflation", "restrictive", "reduce the size of the balance sheet", "taper",
           "tapering", "vigilant", "remove accommodation", "firming", "overheating", "higher for longer",
           "ongoing increases", "forceful", "expeditious", "expedient", "resolute", "strongly committed to returning inflation")
DOVISH = ("ease", "easing", "lower the target", "decrease in the target", "rate cut", "rate cuts",
          "cut rates", "accommodative", "accommodation", "asset purchases", "quantitative easing",
          "patient", "downside risks", "weakening", "slack", "stimulus", "support the economy",
          "lower for longer", "forward guidance", "negative interest", "pause", "moderate the pace",
          "whatever it takes", "backstop", "liquidity facility")
STRESS = ("crisis", "turmoil", "stress", "strains", "volatility", "uncertainty", "uncertain",
          "contagion", "panic", "fragmentation", "sell-off", "selloff", "disorderly", "recession",
          "downturn", "shock", "pandemic", "outbreak", "war", "default", "downgrade", "illiquid",
          "dysfunction", "run on", "bank failure", "tariff", "escalation", "capital outflows")
TAIL = ("unprecedented", "emergency", "exceptional", "extraordinary", "historic", "sharpest",
        "highest since", "lowest since", "largest since", "abrupt", "sudden", "collapse", "plunge",
        "surge", "intervention", "devaluation", "depeg", "abandon", "discontinue", "unscheduled",
        "inter-meeting", "intermeeting")
TOPICS: dict[str, tuple[str, ...]] = {
    "monetary_policy": ("policy rate", "federal funds", "monetary policy", "rate", "committee"),
    "inflation": ("inflation", "prices", "cpi", "pce"),
    "labor": ("employment", "payroll", "unemployment", "labor"),
    "growth": ("growth", "gdp", "output", "activity", "recession"),
    "financial_stability": ("financial stability", "liquidity", "funding", "banks", "credit"),
    "fx": ("exchange rate", "currency", "dollar", "intervention"),
    "balance_sheet": ("asset purchases", "balance sheet", "securities", "reinvest"),
    "geopolitics": ("war", "tariff", "sanctions", "election", "referendum", "trade"),
}
DOC_TYPE_WEIGHT = {"fomc_statement": 1.0, "fomc_minutes": 0.8, "macro_release": 0.8,
                   "cb_speech": 0.6, "testimony": 0.8, "beige_book": 0.5, "landmark": 0.9,
                   "cot": 0.4, "document": 0.5}

# FX quote conventions in the shipped panels: USD per unit for these, units per USD otherwise.
USD_PER_UNIT = {"EUR", "GBP", "AUD", "NZD"}


def asset_key(asset: str) -> str:
    a = asset.upper()
    if a.startswith("UST"):
        return "UST"
    for p in ("CPI", "PCE"):
        if a.startswith(p):
            return p
    return a
