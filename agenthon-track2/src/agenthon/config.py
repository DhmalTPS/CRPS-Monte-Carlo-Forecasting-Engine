"""Configuration: built-in defaults <- configs/default.yaml <- profile yaml <- family yaml.

No magic number lives in the modules: they read `cfg["section"]["key"]`. The built-in DEFAULTS
below are the *frozen* values the image ships with, so a missing or corrupt YAML file degrades to
the validated defaults instead of crashing a card (a config problem must never cost 4.0).

Search order for the config directory:
  1. $AGENTHON_CONFIG_DIR
  2. <repo>/configs (development checkout, resolved relative to this file)
  3. /opt/agenthon/configs (the Docker image)
"""

from __future__ import annotations

import copy
import os
import pathlib
import sys
from typing import Any

DEFAULTS: dict[str, Any] = {
    "runtime": {
        "profile": "final",
        "seed_salt": 20261013,
        "total_seconds": 900,  # our own soft ceiling per unit (hard ceiling 1800 s)
        "numeric_seconds": 60,
        "text_seconds": 420,
    },
    "sampling": {
        "n_default": 2000,
        "n_f4": 4000,
        "n_min": 500,  # never below the card floor; 500 matches M0
        "belief_pool": 6000,  # belief draws the score-aware layer resamples from
        "antithetic": True,
    },
    "m0": {"window": 300, "draws": 500, "override_ratio": 2.0},
    "numerical": {
        "ewma_halflife": 30.0,  # business days
        "long_window": 1260,  # ~5y of steps for the long-run variance
        "lr_omega": 0.5,  # long-run var = (1-w) M0-window var + w long var (prior)
        "scale": 1.0,
        "vol_persistence": 0.985,  # daily AR(1) persistence of conditional variance
        "drift_shrink": 0.0,  # fraction of M0's trailing drift kept (level targets)
        "drift_shrink_return": 0.25,  # log_return targets: shrink toward long-run mean
        "vr_shrink_n0": 40.0,  # pseudo-count of independent s-step blocks for VR shrinkage
        "vr_clip": [0.6, 1.6],
        "df": 12.0,  # prior; fitted vintages use ~Gaussian (walk-forward: fat t-tails hurt the ratio)
        "corr_shrink": 0.15,  # shrink toward the M0 sample correlation / identity blend
        "zlb_enabled": True,
        "zlb_floor": -0.10,  # soft floor for USD yields (% p.a.)
        "macro_trend_halflife": 6.0,  # months, for monthly macro drift
        "macro_trend_weight": 0.7,
        "transfer_proxy_weight": 0.5,
    },
    "analogue": {"enabled": False, "k": 40, "weight": 0.15, "min_history": 750},
    "regime": {"enabled": True, "calm_ratio": 0.8, "stress_ratio": 1.35},
    "tails": {
        "in_unit_backtest": False,  # no robust walk-forward gain (R2_backtest)
        "backtest_stride": 10,
        "backtest_years": 12,
        "n0_scale": 60.0,  # EB pseudo-count for per-asset scale bias
        "n0_df": 120.0,
        "scale_bias_clip": [0.75, 1.35],
        "df_clip": [4.0, 30.0],
    },
    "text": {
        "enabled": True,
        "max_docs": 60,
        "max_chars_per_doc": 120000,
        "chunk_chars": 900,
        "top_k": 12,
        "token_budget_chars": 14000,
        "dedup_jaccard": 0.55,
        "dedup_days": 5,
        "recency_halflife_days": 45.0,
        "retrieval_weights": {"bm25": 1.0, "recency": 0.6, "asset": 0.8, "novelty": 0.3,
                              "redundancy": 0.5, "doc_type": 0.4},
    },
    "llm": {
        "enabled": True,
        "max_calls": 3,  # hard cap of our own (the House allowance is 25/unit)
        "max_tokens": 1800,
        "timeout_seconds": 120,
        "temperature": 0.0,
        "seed": 7,
        "retries": 1,
        "enable_thinking": False,
    },
    "gate": {
        "w0": -1.2,
        "w_retrieval": 1.1,
        "w_confidence": 2.2,
        "w_corroboration": 0.35,
        "w_kappa": 1.4,
        "lexicon_only_cap": 0.25,
        "g_max": 0.85,
    },
    "update": {
        "alpha_shift": 0.25,  # max centre shift per unit gate, in horizon sd (ratio loss is convex)
        "max_text_vol_mult": 2.0,  # hard cap on text-driven widening of any cell
        "beta_vol": 0.30,  # log-scale widening per unit gate for "higher"
        "lambda_kappa": 0.15,  # contradiction widening (never narrows)
        "allow_text_narrowing": False,
        "kappa_star": 0.55,  # switch tilt -> mixture above this contradiction
        "tail_weight_max": 0.12,
        "tail_scale": 2.2,
    },
    "calibration": {"enabled": False, "shrink_n0": 200.0},  # M10: PIT maps hurt OOS (0.964 -> 0.977)
    "decision": {
        "enabled": True,
        "gamma": 0.5,  # prior from the ratio-loss simulation (no data); vintages carry fitted values
        "lambda": 0.4,  # prior: hedge harder when nothing has been fitted (walk-forward 2003-08)
        "robust_mix": 0.30,  # outcome belief = (1-r)*P + r*M0 when evaluating actions
        "gamma_grid": [0.0, 0.5, 1.0, 1.25],
        "lambda_grid": [0.0, 0.2, 0.3],
        "min_sd_ratio": 0.60,  # never submit narrower than 60% of the honest belief sd
        "optimise": False,  # per-card self-optimisation self-confirms; fixed/policy action
        "eval_outcomes": 3000,
    },
    "policy": {"enabled": True, "pessimism": 2.0},  # LCB: only pessimistic policies beat default OOS
    "output": {"max_draws": 20000},
}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_dir() -> pathlib.Path | None:
    env = os.environ.get("AGENTHON_CONFIG_DIR")
    cands = [pathlib.Path(env)] if env else []
    cands.append(pathlib.Path(__file__).resolve().parents[2] / "configs")
    cands.append(pathlib.Path("/opt/agenthon/configs"))
    for c in cands:
        if (c / "default.yaml").is_file():
            return c
    return None


def _read_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # corrupt/missing yaml must not cost a card
        print(f"[agenthon] config {path.name} ignored: {type(exc).__name__}", file=sys.stderr)
        return {}


def load_config(profile: str | None = None, family: str | None = None,
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    d = config_dir()
    if d is not None:
        cfg = _deep_merge(cfg, _read_yaml(d / "default.yaml"))
        prof = profile or os.environ.get("AGENTHON_PROFILE") or cfg["runtime"].get("profile")
        if prof and (d / f"{prof}.yaml").is_file():
            cfg = _deep_merge(cfg, _read_yaml(d / f"{prof}.yaml"))
        if family:
            fam = family.lower().replace("t2-", "")
            fpath = d / "families" / f"{fam}.yaml"
            if fpath.is_file():
                cfg = _deep_merge(cfg, _read_yaml(fpath))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
