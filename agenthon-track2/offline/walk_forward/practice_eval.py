"""Evaluate text rungs (and the full system) on the 103 PRACTICE units, which carry dated corpora.

Realised outcomes are reconstructed from the published history (the h-th panel row after the
as-of; sum of log(1+r) for log_return). The organizers state that most practice answers are
readable from sibling panels; we therefore use them strictly as an EVALUATION set (never baked
into the image, never used to fit a parameter) and treat practice numbers as a secondary signal
next to the walk-forward protocol. Monthly macro and EM-transfer cards are skipped when their
outcome is not in the reconstructed history.

delta_text(card) = S(numeric-only) - S(with text rung); > 0 means text helped.

    python offline/walk_forward/practice_eval.py --rung M5_gate --units <units-dir>
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
DEFAULT_UNITS = ROOT.parent / "_reference" / "track2-forecasting-public" / "units"

TEXT_RUNGS = {
    "M3_numeric": {"text": {"enabled": False}},
    "M4_text": {"gate": {"w_kappa": 0.0}, "update": {"tail_weight_max": 0.0, "kappa_star": 9.0,
                                                     "lambda_kappa": 0.0}, "families": {}},
    "M5_gate": {"update": {"tail_weight_max": 0.0, "kappa_star": 9.0, "lambda_kappa": 0.0}},
    "M6_contradiction": {"update": {"tail_weight_max": 0.0}},
    "M7_distribution_update": {},
    "M8_dependence": {},
    "M11_full": {},
}


def _history() -> dict[str, pd.Series]:
    from offline.walk_forward.generate_cutoffs import load_history

    return load_history()


def realised_for(card, hist: dict[str, pd.Series]) -> np.ndarray | None:
    from offline.walk_forward.generate_cutoffs import realised

    if card.target_frequency == "monthly":
        return None
    ys = []
    for a in sorted(card.assets):
        s = hist.get(a)
        if s is None:
            return None
        for h in sorted(card.horizons):
            r = realised(s, pd.Timestamp(card.asof), int(h), card.target_type)
            if r is None:
                return None
            ys.append(r[0])
    return np.asarray(ys)


def evaluate_units(units: pathlib.Path, rung: str, extra: dict | None = None) -> pd.DataFrame:
    from agenthon.card import parse_card
    from agenthon.config import _deep_merge, load_config
    from agenthon.data import point_in_time_filter
    from agenthon.decision import m0_scoring as ms
    from agenthon.main import forecast_from_pit

    hist = _history()
    rows = []
    for u in sorted(units.iterdir()):
        if not (u / "card.toml").is_file() or u.name.startswith("t2-EXAMPLE"):
            continue
        card = parse_card(u / "card.toml")
        y = realised_for(card, hist)
        if y is None:
            continue
        cfg = load_config("development", card.family)
        cfg = _deep_merge(cfg, {"llm": {"enabled": False}})
        cfg = _deep_merge(cfg, TEXT_RUNGS.get(rung, {}))
        if extra:
            cfg = _deep_merge(cfg, extra)
        cfg["_vintage"] = "practice-eval"
        pit = point_in_time_filter(card, u / "panels", u / "text", card.asof, cfg)
        res = forecast_from_pit(card, pit, card.asof, cfg)
        w = card.effective_weights()
        joint = card.n_cells > 1
        c0 = ms.components(res.diagnostics["m0_draws"], y[None], card.tail_levels, joint)
        sc = float(ms.normalised(ms.components(res.draws, y[None], card.tail_levels, joint), c0, w)[0])
        rows.append({"unit": card.unit_id, "family": card.family, "score": sc,
                     "gate": res.diagnostics["text"]["gate"], "gamma": res.diagnostics["gamma"],
                     "source": res.diagnostics["text"]["source"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="M11_full")
    ap.add_argument("--units", type=pathlib.Path, default=DEFAULT_UNITS)
    ap.add_argument("--overrides", default="{}")
    a = ap.parse_args()
    extra = json.loads(a.overrides)
    base = evaluate_units(a.units, "M3_numeric", extra)
    cur = evaluate_units(a.units, a.rung, extra) if a.rung != "M3_numeric" else base
    m = base.merge(cur, on=["unit", "family"], suffixes=("_p0", "_p1"))
    m["delta_text"] = m["score_p0"] - m["score_p1"]
    out = ROOT / "offline/reports/tables" / f"practice_{a.rung}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(m.groupby("family")[["score_p0", "score_p1", "delta_text", "gate_p1"]].mean().round(4).to_string())
    print(f"{a.rung}: n={len(m)} mean score P0 {m.score_p0.mean():.4f} P1 {m.score_p1.mean():.4f} "
          f"delta_text {m.delta_text.mean():+.4f} (helped on {(m.delta_text > 0).mean():.0%} of cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
