"""Walk-forward evaluation of one model variant over the synthetic cards.

For every card: build the point-in-time slice of the reconstructed history (every row <= as-of),
run the PRODUCTION pipeline (agenthon.main.forecast_from_pit) with the variant's config overrides,
and score against the realised outcome, normalised by the exact M0 replica, i.e. the leaderboard
statistic. Because y is known offline, every gamma action is also scored on the same belief pool
(full-information rewards for the policy learner).

    python offline/walk_forward/run.py --name M3_default --overrides '{"decision":{"optimise":false}}'
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agenthon.config import _deep_merge, load_config  # noqa: E402
from agenthon.data.availability import asset_class  # noqa: E402
from agenthon.decision import m0_scoring as ms  # noqa: E402
from agenthon.decision.score_aware import _m0_weight, resample  # noqa: E402
from agenthon.types import AssetHistory, Card, PointInTimeData  # noqa: E402

GAMMAS = (0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0)
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4)
MIN_SD_RATIO = 0.6  # production floor (decision.min_sd_ratio)
_HIST: dict[str, tuple[pd.Series, str]] = {}


def _init() -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    from offline.walk_forward.generate_cutoffs import DATA, PANEL_OF

    for cls, f in PANEL_OF.items():
        df = pd.read_parquet(DATA / f)
        df["date"] = pd.to_datetime(df["date"])
        for a, g in df.groupby("asset"):
            _HIST[str(a)] = (g.set_index("date")["value"].astype(float).sort_index(), cls)


def make_card(c: dict) -> Card:
    return Card(unit_id=c["unit_id"], asof=c["asof"], family=c["family"], assets=tuple(c["assets"]),
                horizons=tuple(c["horizons"]), target_type=c["target_type"], target_frequency="daily",
                value_unit="", n_draws_min=200, weights=(0.5, 0.3, 0.2),
                tail_levels=(0.01, 0.05, 0.95, 0.99), joint="variogram", title=c["template"])


def make_pit(c: dict) -> PointInTimeData:
    asof = pd.Timestamp(c["asof"])
    classes = {_HIST[a][1] for a in c["assets"]}
    hist = {}
    for a, (s, cls) in _HIST.items():
        if cls not in classes and a not in c["assets"]:
            continue
        sl = s[s.index <= asof]
        if len(sl) > 30:
            pid = {"rates": "rates_daily", "fx": "g10_fx_daily", "factor": "factors_daily"}[cls]
            hist[a] = AssetHistory(a, pid, sl, False, asset_class(a, pid))
    return PointInTimeData(asof=c["asof"], histories=hist, docs=[])


def evaluate(args: tuple[dict, dict]) -> dict:
    from agenthon.main import forecast_from_pit

    c, overrides = args
    overrides = json.loads(json.dumps(overrides))  # never mutate the shared (pickled-chunk) dict
    card = make_card(c)
    cfg = load_config("development", card.family)
    label = "walk-forward"
    if overrides.pop("_vintaged", False):
        from agenthon.resources import vintage_overrides

        vo, label = vintage_overrides(c["asof"], card.family)  # exactly the production selection rule
        cfg = _deep_merge(cfg, vo)
    cfg = _deep_merge(cfg, {"text": {"enabled": False}, "llm": {"enabled": False}})
    cfg = _deep_merge(cfg, overrides)
    cfg["_vintage"] = label
    t = time.time()
    try:
        res = forecast_from_pit(card, make_pit(c), c["asof"], cfg)
    except Exception as exc:
        return {"unit_id": c["unit_id"], "error": f"{type(exc).__name__}: {exc}"}
    y = np.asarray(c["y"], dtype=float)[None, :]
    w = card.effective_weights()
    joint = card.n_cells > 1
    m0 = res.diagnostics["m0_draws"]
    c0 = ms.components(m0, y, card.tail_levels, joint)
    sub = float(ms.normalised(ms.components(res.draws, y, card.tail_levels, joint), c0, w)[0])
    pool = res.diagnostics["pool"]
    rng = np.random.default_rng(zlib.crc32(c["unit_id"].encode()))
    bw = _m0_weight(pool, m0, w, card.tail_levels, joint)
    per_gamma = {}
    n = 2000
    for g in GAMMAS:
        q, _ = resample(pool, bw, g, n, rng)
        if g > 0 and np.any(q.std(axis=0) < MIN_SD_RATIO * pool.std(axis=0)):
            for lam in LAMBDAS:
                per_gamma[f"g{g:.2f}" if lam == 0 else f"g{g:.2f}_l{lam:.2f}"] = float("nan")
            continue
        for lam in LAMBDAS:
            k = int(round(lam * n))
            qq = q if k == 0 else np.vstack([q[: n - k], m0[rng.integers(0, len(m0), k)]])
            key = f"g{g:.2f}" if lam == 0 else f"g{g:.2f}_l{lam:.2f}"
            per_gamma[key] = float(ms.normalised(ms.components(qq, y, card.tail_levels, joint), c0, w)[0])
    sd_ratio = float(np.mean(pool.std(axis=0) / np.maximum(m0.std(axis=0), 1e-12)))
    pit = [float(np.mean(pool[:, j] <= y[0, j])) for j in range(pool.shape[1])]
    classes = [res.diagnostics["assets"][cell.asset]["class"] for cell in res.grid.cells]
    steps = [cell.steps for cell in res.grid.cells]
    z = float(np.mean((y[0] - m0.mean(axis=0)) / np.maximum(m0.std(axis=0), 1e-12)))
    return {"unit_id": c["unit_id"], "template": c["template"], "family": c["family"],
            "asof": c["asof"], "target_end": c["target_end"], "cells": card.n_cells,
            "score": sub, "gamma": res.diagnostics.get("gamma"), "sd_ratio_vs_m0": sd_ratio,
            "z_m0": z, "seconds": time.time() - t, "vintage": label, "lambda": res.diagnostics.get("lambda"),
            "pit": json.dumps(pit), "classes": json.dumps(classes), "steps": json.dumps(steps),
            "x": json.dumps([float(v) for v in res.diagnostics["context"]]), **per_gamma}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=pathlib.Path, default=ROOT / "offline/experiments/configs/cards.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--since", default="")
    a = ap.parse_args()
    cards = json.loads(a.cards.read_text())
    if a.since:
        cards = [c for c in cards if c["asof"] >= a.since]
    if a.limit:
        cards = cards[:: max(1, len(cards) // a.limit)][: a.limit]
    ov = json.loads(a.overrides)
    t = time.time()
    rows = []
    with cf.ProcessPoolExecutor(a.jobs, initializer=_init) as ex:
        for r in ex.map(evaluate, [(c, ov) for c in cards], chunksize=8):
            rows.append(r)
    df = pd.DataFrame(rows)
    out = ROOT / "offline/experiments/runs" / f"{a.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    err = df["error"].notna().sum() if "error" in df else 0
    ok = df[df.get("error").isna()] if "error" in df else df
    print(f"{a.name}: {len(df)} cards, {err} errors, mean score {ok['score'].mean():.4f} "
          f"(median {ok['score'].median():.4f}) in {time.time() - t:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
