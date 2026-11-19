"""Summaries of walk-forward runs with block-bootstrap confidence intervals.

Cards whose as-of dates are close share outcome windows (overlap), so the bootstrap resamples
calendar QUARTERS of as-of dates, not cards: honest error bars under cross-card dependence.

    python offline/walk_forward/aggregate.py M1 M3 M11 [--split 2015-01-01] [--by family]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "offline/experiments/runs"


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RUNS / f"{name}.parquet")
    if "error" in df:
        df = df[df["error"].isna()]
    df["quarter"] = pd.to_datetime(df["asof"]).dt.to_period("Q").astype(str)
    return df


def block_ci(df: pd.DataFrame, col: str = "score", n: int = 400, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    g = df.groupby("quarter")[col].agg(["sum", "count"])
    if len(g) < 3:
        m = float(df[col].mean())
        return m, m, m
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g))
        stats.append(sums[i].sum() / cnts[i].sum())
    return float(df[col].mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired_ci(a: pd.DataFrame, b: pd.DataFrame, n: int = 400) -> tuple[float, float, float]:
    m = a[["unit_id", "score", "quarter"]].merge(b[["unit_id", "score"]], on="unit_id", suffixes=("_a", "_b"))
    m["d"] = m["score_a"] - m["score_b"]
    return block_ci(m, "d", n)


def table(names: list[str], by: str | None, split: str | None) -> pd.DataFrame:
    rows = []
    for nm in names:
        df = load(nm)
        parts = [("all", df)]
        if split:
            parts += [(f"<{split}", df[df["asof"] < split]), (f">={split}", df[df["asof"] >= split])]
        for lab, d in parts:
            if by:
                for k, g in d.groupby(by):
                    mean, lo, hi = block_ci(g)
                    rows.append({"run": nm, "slice": lab, by: k, "n": len(g), "mean": mean, "lo": lo, "hi": hi})
            else:
                mean, lo, hi = block_ci(d)
                rows.append({"run": nm, "slice": lab, "n": len(d), "mean": mean, "lo": lo, "hi": hi,
                             **{c: d[c].mean() for c in d.columns if c.startswith("g0.") or c.startswith("g1")}})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--by", default=None)
    ap.add_argument("--split", default="2015-01-01")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    t = table(a.names, a.by, a.split)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))
    if a.save:
        out = ROOT / "offline/reports/tables" / f"{a.save}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
