"""Full-information action table on the PRACTICE units (event-selected cards), per family.

Belief = the fitted 2024 vintage (numeric only, text off). For every (gamma, lambda) action the
normalised score on the reconstructed outcome is recorded, exactly as run.py does on synthetic
cards. Used ONLY to set family-level action overrides in the 2024 vintage (the outcomes reach 2024,
so earlier vintages must not see them).

    python offline/walk_forward/practice_actions.py
"""

from __future__ import annotations

import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agenthon.card import parse_card  # noqa: E402
from agenthon.config import _deep_merge, load_config  # noqa: E402
from agenthon.data import point_in_time_filter  # noqa: E402
from agenthon.decision import m0_scoring as ms  # noqa: E402
from agenthon.decision.score_aware import _m0_weight, mix_m0, resample  # noqa: E402
from agenthon.main import forecast_from_pit  # noqa: E402
from agenthon.resources import load_artifact  # noqa: E402
from offline.walk_forward.practice_eval import DEFAULT_UNITS, _history, realised_for  # noqa: E402

GAMMAS = (0.0, 0.5, 0.8, 1.0, 1.25)
LAMBDAS = (0.0, 0.2, 0.3, 0.4, 0.6, 0.8)


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--belief", default="{}", help="extra config overrides for the belief")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    extra = json.loads(a.belief)
    hist = _history()
    v24 = [v for v in load_artifact("params_vintages.json")["vintages"] if v["fit_cutoff"] == "2024-12-31"][0]
    rows = []
    for u in sorted(DEFAULT_UNITS.iterdir()):
        if not (u / "card.toml").is_file() or u.name.startswith("t2-EXAMPLE"):
            continue
        card = parse_card(u / "card.toml")
        if a.families and card.family not in a.families.split(","):
            continue
        y = realised_for(card, hist)
        if y is None:
            continue
        cfg = _deep_merge(load_config("development", card.family), v24["overrides"])
        cfg = _deep_merge(cfg, {"llm": {"enabled": False}, "text": {"enabled": False}})
        cfg = _deep_merge(cfg, extra)
        cfg["_vintage"] = "practice"
        res = forecast_from_pit(card, point_in_time_filter(card, u / "panels", u / "text", card.asof, cfg),
                                card.asof, cfg)
        w, joint = card.effective_weights(), card.n_cells > 1
        m0, pool = res.diagnostics["m0_draws"], res.diagnostics["pool"]
        c0 = ms.components(m0, y[None], card.tail_levels, joint)
        bw = _m0_weight(pool, m0, w, card.tail_levels, joint)
        rng = np.random.default_rng(zlib.crc32(card.unit_id.encode()))
        row = {"unit": card.unit_id, "family": card.family}
        for g in GAMMAS:
            q, _ = resample(pool, bw, g, 2000, rng)
            if g > 0 and np.any(q.std(0) < 0.6 * pool.std(0)):
                continue
            for lam in LAMBDAS:
                qq = mix_m0(q, m0, lam, rng)
                row[f"g{g:.2f}_l{lam:.2f}"] = float(ms.normalised(ms.components(qq, y[None], card.tail_levels, joint), c0, w)[0])
        rows.append(row)
    d = pd.DataFrame(rows)
    out = ROOT / f"offline/reports/tables/practice_actions{a.tag}.csv"
    d.to_csv(out, index=False)
    cols = [c for c in d.columns if c.startswith("g")]
    pd.set_option("display.width", 250)
    for fam, g in d.groupby("family"):
        m = g[cols].mean().sort_values()
        print(fam, len(g), "best:", ", ".join(f"{k}={v:.3f}" for k, v in m.head(4).items()),
              "| g1.00_l0.30:", round(g["g1.00_l0.30"].mean(), 3), "| g0.00_l0.00:", round(g["g0.00_l0.00"].mean(), 3))
    m = d[cols].mean().sort_values()
    print("ALL best:", ", ".join(f"{k}={v:.3f}" for k, v in m.head(5).items()), "| vintage action g1.00_l0.30:",
          round(d["g1.00_l0.30"].mean(), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
