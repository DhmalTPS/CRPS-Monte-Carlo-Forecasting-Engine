"""One-factor-at-a-time tuning of the belief model around a base, scored by the BEST fixed action
(gamma, lambda) chosen on the TRAIN split and reported on TRAIN and TEST separately.

Honesty rules: the action and the variant are selected on train (as-of < split, outcome known
before split); test numbers are reported, never used to choose. The search space is fixed here,
before running (no open-ended search).

    python offline/experiments/tune.py --split 2015-01-01 --jobs 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = {"decision": {"enabled": False}, "calibration": {"enabled": False}, "policy": {"enabled": False},
        "tails": {"in_unit_backtest": False},
        "numerical": {"df": 1000.0, "drift_shrink": 0.0, "drift_shrink_return": 0.0}}
VARIANTS = {
    "base": {},
    "df7": {"numerical": {"df": 7.0}},
    "df12": {"numerical": {"df": 12.0}},
    "backtest": {"tails": {"in_unit_backtest": True}},
    "omega0": {"numerical": {"lr_omega": 0.0}},
    "omega1": {"numerical": {"lr_omega": 1.0}},
    "hl15": {"numerical": {"ewma_halflife": 15.0}},
    "hl60": {"numerical": {"ewma_halflife": 60.0}},
    "phi0.97": {"numerical": {"vol_persistence": 0.97}},
    "phi0.995": {"numerical": {"vol_persistence": 0.995}},
    "scale0.9": {"numerical": {"scale": 0.9}},
    "scale1.1": {"numerical": {"scale": 1.1}},
    "novr": {"numerical": {"vr_clip": [1.0, 1.0]}},
    "noret_drift": {"numerical": {"drift_shrink_return": 1.0}},
}


def merge(a: dict, b: dict) -> dict:
    out = json.loads(json.dumps(a))
    for k, v in b.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def summarise(name: str, split: str) -> dict:
    d = pd.read_parquet(ROOT / "offline/experiments/runs" / f"{name}.parquet")
    d = d[d.get("error").isna()] if "error" in d else d
    cols = [c for c in d.columns if c.startswith("g") and c[1].isdigit()]
    tr = d[(d["asof"] < split) & (d["target_end"] < split)]
    te = d[d["asof"] >= split]
    # an action is admissible only where it passed the sd floor on EVERY card (NaN = refused);
    # refused cells fall back to the honest belief in production, so score them that way here
    trf = tr[cols].apply(lambda c: c.fillna(tr["g0.00"]))
    tef = te[cols].apply(lambda c: c.fillna(te["g0.00"]))
    tr, te = tr.assign(**trf), te.assign(**tef)
    best = tr[cols].mean().idxmin()
    return {"variant": name, "best_action": best, "train": tr[best].mean(), "test": te[best].mean(),
            "test_honest": te["g0.00"].mean(), "train_honest": tr["g0.00"].mean(),
            "test_bestaction_on_test": te[cols].mean().min()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2015-01-01")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--prefix", default="T_")
    ap.add_argument("--base", default="")
    a = ap.parse_args()
    base = merge(BASE, json.loads(a.base)) if a.base else BASE
    rows = []
    for v, ov in VARIANTS.items():
        if a.only and v not in a.only.split(","):
            continue
        name = a.prefix + v
        subprocess.run([sys.executable, str(ROOT / "offline/walk_forward/run.py"), "--name", name,
                        "--jobs", str(a.jobs), "--overrides", json.dumps(merge(base, ov))], check=True)
        rows.append(summarise(name, a.split))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows).sort_values("train")
    print(t.round(4).to_string(index=False))
    out = ROOT / "offline/reports/tables" / f"{a.prefix}tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
