"""M12 stress slice: scores restricted to crisis windows, plus the worst-decile view.

    python offline/walk_forward/stress.py M11_full [M1_random_walk ...]
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
WINDOWS = [("GFC", "2008-06-01", "2009-06-30"), ("taper", "2013-04-01", "2013-12-31"),
           ("covid", "2020-01-15", "2020-07-31"), ("hikes2022", "2022-01-01", "2022-12-31")]


def main() -> int:
    rows = []
    for name in sys.argv[1:]:
        d = pd.read_parquet(ROOT / "offline/experiments/runs" / f"{name}.parquet")
        for lab, a, b in WINDOWS:
            s = d[(d["asof"] >= a) & (d["asof"] <= b)]
            rows.append({"run": name, "window": lab, "n": len(s), "mean": s["score"].mean(),
                         "p90": s["score"].quantile(0.9), "frac_clip": (s["score"] >= 3.999).mean()})
        rows.append({"run": name, "window": "worst_decile_of_M0_z", "n": int(len(d) * 0.1),
                     "mean": d.loc[d["z_m0"].abs().nlargest(int(len(d) * 0.1)).index, "score"].mean(),
                     "p90": float("nan"), "frac_clip": float("nan")})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    t.to_csv(ROOT / "offline/reports/tables/stress.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
