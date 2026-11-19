"""Fit L22 PIT recalibration maps per (asset class | horizon bucket) from walk-forward PITs.

PITs are of the HONEST belief pool (before the score-aware layer), recorded by run.py. Only cards
whose outcome is known by the vintage cutoff are used. Maps are shrunk toward the identity with
pseudo-count n0 (calibration.shrink_n0). The ablation (M10) decides whether they ship.

    python offline/experiments/fit_calibration.py --run M11_full --cutoffs 2014-12-31,2024-12-31 [--write]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agenthon.calibration import horizon_bucket  # noqa: E402
from agenthon.calibration.maps import fit_inverse_pit  # noqa: E402


def pit_table(run: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "offline/experiments/runs" / f"{run}.parquet")
    rows = []
    for _, r in d.iterrows():
        for p, c, s in zip(json.loads(r["pit"]), json.loads(r["classes"]), json.loads(r["steps"])):
            rows.append({"asof": r["asof"], "target_end": r["target_end"], "key": f"{c}|{horizon_bucket(s)}",
                         "pit": p})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="M11_full")
    ap.add_argument("--cutoffs", default="2014-12-31,2024-12-31")
    ap.add_argument("--n0", type=float, default=200.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t = pit_table(a.run)
    vint = []
    for c in a.cutoffs.split(","):
        sub = t[t["target_end"] <= c]
        maps = {}
        for k, g in sub.groupby("key"):
            maps[k] = fit_inverse_pit(g["pit"].to_numpy(), a.n0)
            q = np.quantile(g["pit"], [0.05, 0.5, 0.95])
            print(f"{c} {k:18s} n={len(g):5d} PIT q05/q50/q95 = {q.round(3).tolist()}")
        vint.append({"fit_cutoff": c, "n0": a.n0, "maps": maps})
    if a.write:
        p = ROOT / "src/agenthon/resources/calibration_vintages.json"
        p.write_text(json.dumps({"description": "PIT maps (G^-1 knots on maps.U_GRID) per class|bucket",
                                 "vintages": vint}), encoding="utf-8")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
