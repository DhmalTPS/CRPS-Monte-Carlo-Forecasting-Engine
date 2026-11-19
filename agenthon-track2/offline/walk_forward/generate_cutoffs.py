"""Synthetic walk-forward cards on the reconstructed 2000-2024 panels (the primary validation signal).

Card templates mirror the practice roster's shapes (measured: 60/104 single-cell, horizons
21/63/126/189, 88 level vs 16 log_return, UST curves, G10 baskets, mixed rates+FX, factor pairs).
For each template and a grid of as-of dates (every `stride` business days), the realised target
is read from the history: the value at the h-th panel row after the as-of (level), or the sum of
log(1 + r) over the next h rows (log_return).

Every card records `target_end` (last date its outcome needs) so a fitting set can be restricted to
outcomes known by a vintage's fit cutoff.

    python offline/walk_forward/generate_cutoffs.py --out offline/experiments/configs/cards.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "offline" / "data"

TEMPLATES = [
    # name, family, target_type, assets, horizons, weight
    ("ust_single_long", "T2-F1", "level", [["UST_2Y"], ["UST_10Y"]], [126, 189], 3),
    ("fx_single_long", "T2-F1", "level", [["EUR"], ["JPY"], ["AUD"], ["CAD"], ["CHF"], ["GBP"]], [126, 189], 3),
    ("factor_single_long", "T2-F1", "log_return", [["MKT"], ["MOM"]], [127], 1),
    ("ust_single_mid", "T2-F2", "level", [["UST_2Y"], ["UST_10Y"]], [63], 3),
    ("fx_single_mid", "T2-F2", "level", [["EUR"], ["JPY"], ["AUD"], ["CAD"], ["GBP"], ["NOK"]], [64], 3),
    ("ust_curve", "T2-F3", "level", [["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]], [63, 126], 2),
    ("ust_curve_short", "T2-F3", "level", [["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]], [21, 63], 1),
    ("fx_basket", "T2-F3", "level", [["GBP", "EUR", "CHF", "JPY"], ["NOK", "CAD", "EUR", "SEK"],
                                     ["JPY", "CHF", "AUD", "NZD"]], [21, 63], 2),
    ("mixed", "T2-F3", "level", [["UST_10Y", "JPY"], ["UST_2Y", "EUR", "GBP", "JPY"]], [21, 63], 2),
    ("factor_pair", "T2-F3", "log_return", [["MKT", "HML"], ["HML", "MOM"]], [21, 63], 1),
    ("ust_single_short", "T2-F4", "level", [["UST_2Y"], ["UST_10Y"]], [21], 3),
    ("fx_single_short", "T2-F4", "level", [["AUD"], ["JPY"], ["GBP"], ["EUR"], ["CHF"], ["NOK"]], [21], 3),
    ("factor_single_short", "T2-F4", "log_return", [["MKT"], ["HML"], ["MOM"]], [21], 3),
]
PANEL_OF = {"rates": "rates_daily_history.parquet", "fx": "g10_fx_daily_history.parquet",
            "factor": "factors_daily_history.parquet"}


def load_history() -> dict[str, pd.Series]:
    out = {}
    for f in PANEL_OF.values():
        df = pd.read_parquet(DATA / f)
        df["date"] = pd.to_datetime(df["date"])
        for a, g in df.groupby("asset"):
            out[str(a)] = g.set_index("date")["value"].astype(float).sort_index()
    return out


def realised(s: pd.Series, asof: pd.Timestamp, h: int, target_type: str) -> tuple[float, str] | None:
    pos = s.index.searchsorted(asof, side="right")  # first row after the as-of
    end = pos + h - 1
    if pos == 0 or end >= len(s):
        return None
    if target_type == "log_return":
        r = s.to_numpy()[pos:end + 1]
        return float(np.sum(np.log1p(r))), str(s.index[end].date())
    return float(s.iloc[end]), str(s.index[end].date())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "offline/experiments/configs/cards.json")
    ap.add_argument("--stride", type=int, default=21)
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--per-template", type=int, default=260)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    hist = load_history()
    rng = np.random.default_rng(a.seed)
    master = hist["UST_10Y"].index
    dates = master[master >= pd.Timestamp(a.start)][::a.stride]
    cards = []
    for name, fam, tt, groups, hz, w in TEMPLATES:
        n_want = a.per_template * w // 2
        cand = [(g, d) for g in groups for d in dates]
        idx = rng.permutation(len(cand))
        made = 0
        for i in idx:
            g, d = cand[i]
            if any(x not in hist for x in g):
                continue
            ys, ends, ok = [], [], True
            for asset in sorted(g):
                s = hist[asset]
                if s.index[0] > d - pd.Timedelta(days=900):
                    ok = False
                    break
                for h in sorted(hz):
                    r = realised(s, d, h, tt)
                    if r is None:
                        ok = False
                        break
                    ys.append(r[0])
                    ends.append(r[1])
            if not ok:
                continue
            asof = str(d.date())
            uid = f"wf-{name}-{'-'.join(sorted(g))}-{asof}"
            cards.append({"unit_id": uid, "template": name, "family": fam, "target_type": tt,
                          "assets": list(g), "horizons": list(hz), "asof": asof,
                          "y": ys, "target_end": max(ends), "hash": zlib.crc32(uid.encode())})
            made += 1
            if made >= n_want:
                break
    cards.sort(key=lambda c: (c["asof"], c["unit_id"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(cards))
    print(f"{len(cards)} cards -> {a.out}")
    print(pd.Series([c["template"] for c in cards]).value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
