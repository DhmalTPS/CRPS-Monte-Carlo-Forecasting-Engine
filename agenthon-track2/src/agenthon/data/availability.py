"""L1 — the availability-time operator Phi_avail(D; t) = {x in D : t_available(x) <= t}.

Three timestamps are distinct: event time, publication time and availability time. For panels we
only have the row date (which is the availability date for daily market closes; monthly macro
panels are already staged as the vintage published on the card's as-of). For documents we use
the corpus index timestamp, which the organizers define as the PUBLIC RELEASE date (minutes and COT
are dated by release, not by meeting). The staging gates already enforce both; this is
defence-in-depth against our own preprocessing widening the information set.
"""

from __future__ import annotations

import re

import pandas as pd

G10 = {"EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK", "DKK"}
FACTORS = {"MKT", "SMB", "HML", "MOM", "BAB", "QMJ", "RMW", "CMA"}
MACRO_PREFIX = ("CPI", "PCE", "NFP", "UNRATE", "GDP", "PAYEMS")
_DATE_IN_NAME = re.compile(r"(19|20)\d{2}[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")


def asset_class(asset: str, panel: str) -> str:
    a = asset.upper()
    p = (panel or "").lower()
    if "em_" in p or "transfer" in p:
        return "em_fx"
    if a.startswith("UST") or "rates" in p:
        return "rates"
    if a in G10 or "fx" in p:
        return "fx"
    if a in FACTORS or "factor" in p:
        return "factor"
    if a.startswith(MACRO_PREFIX) or "macro" in p:
        return "macro"
    if len(a) == 3 and a.isalpha():
        return "em_fx"  # an unseen 3-letter currency code (e.g. sealed EM FX)
    return "other"


def filter_series(s: pd.Series, asof: str) -> tuple[pd.Series, int]:
    cut = pd.Timestamp(asof)
    keep = s.index <= cut
    return s[keep], int((~keep).sum())


def date_from_filename(name: str) -> str | None:
    m = _DATE_IN_NAME.search(name)
    if not m:
        return None
    digits = re.sub(r"[-_]", "", m.group(0))
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def doc_available(timestamp: str | None, asof: str) -> bool:
    if not timestamp:
        return False  # undated documents are excluded: availability cannot be established
    return str(timestamp)[:10] <= asof
