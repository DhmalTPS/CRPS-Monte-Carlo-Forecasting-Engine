"""Train the safe contextual policy on walk-forward full-information rewards; evaluate out of sample.

Reward design for noisy finance (see policy/bandit.py): the target is the ADVANTAGE of an action
over the default action on the SAME card and the SAME outcome, adv_a = S_default(y) - S_a(y).
The card's own difficulty (a 4-sigma surprise scores badly under every action) cancels, which is
what makes a small linear model learnable at all. Selection is pessimistic (LCB > 0 required).

Protocol: fit on cards whose outcome is known before `--split`; report fixed-default vs policy on
cards with as-of >= split. Vintages: `--vintage-cutoff` restricts fitting to outcomes <= cutoff
and writes src/agenthon/resources/policy_vintages.json entries.

    python offline/rl/train_policy.py --run R2_base --split 2015-01-01
    python offline/rl/train_policy.py --run R2_base --write --cutoffs 2014-12-31,2024-12-31
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

from agenthon.policy.actions import ACTIONS, action_index, action_key  # noqa: E402
from agenthon.policy.bandit import SafePolicy  # noqa: E402


def load(run: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    d = pd.read_parquet(ROOT / "offline/experiments/runs" / f"{run}.parquet")
    d = d[d["error"].isna()] if "error" in d else d
    X = np.array([json.loads(x) for x in d["x"]])
    S = np.full((len(d), len(ACTIONS)), np.nan)
    for i, (g, l) in enumerate(ACTIONS):
        k = action_key(g, l)
        if k in d:
            S[:, i] = d[k].to_numpy()
    honest = d["g0.00"].to_numpy()
    S = np.where(np.isnan(S), honest[:, None], S)  # refused by the sd floor -> honest belief
    return d.reset_index(drop=True), X, S, 0


def fit(X: np.ndarray, S: np.ndarray, default: int, lam: float) -> SafePolicy:
    adv = S[:, [default]] - S  # > 0: action better than default
    pol = SafePolicy(len(ACTIONS), X.shape[1], default, lam)
    pol.fit(X, adv)
    return pol


def evaluate(pol: SafePolicy, X: np.ndarray, S: np.ndarray, pessimism: float) -> tuple[float, float, float]:
    ch = np.array([pol.select(x, pessimism)[0] for x in X])
    pol_score = S[np.arange(len(S)), ch].mean()
    return float(S[:, pol.default].mean()), float(pol_score), float((ch != pol.default).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R2_base")
    ap.add_argument("--split", default="2015-01-01")
    ap.add_argument("--gamma", type=float, default=1.25)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--ridge", type=float, default=50.0)
    ap.add_argument("--pessimism", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--cutoffs", default="2014-12-31,2024-12-31")
    a = ap.parse_args()
    d, X, S, _ = load(a.run)
    default = action_index(a.gamma, a.lam)
    tr = ((d["asof"] < a.split) & (d["target_end"] < a.split)).to_numpy()
    te = (d["asof"] >= a.split).to_numpy()
    rows = []
    for pess in (0.0, 0.5, 1.0, 2.0):
        pol = fit(X[tr], S[tr], default, a.ridge)
        base, ps, frac = evaluate(pol, X[te], S[te], pess)
        rows.append({"pessimism": pess, "test_default": base, "test_policy": ps, "frac_deviate": frac,
                     "train_policy": evaluate(pol, X[tr], S[tr], pess)[1]})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    out = ROOT / "offline/reports/tables/policy_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    if a.write:
        vint = []
        for c in a.cutoffs.split(","):
            m = (d["target_end"] <= c).to_numpy()
            pol = fit(X[m], S[m], default, a.ridge)
            vint.append({"fit_cutoff": c, "n_cards": int(m.sum()), "default_action": [a.gamma, a.lam],
                         "pessimism": a.pessimism, "policy": pol.to_dict()})
        p = ROOT / "src/agenthon/resources/policy_vintages.json"
        p.write_text(json.dumps({"description": "SafePolicy per vintage; see offline/rl/train_policy.py",
                                 "features": "agenthon.policy.features.NAMES", "vintages": vint}))
        print(f"wrote {p} ({len(vint)} vintages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
