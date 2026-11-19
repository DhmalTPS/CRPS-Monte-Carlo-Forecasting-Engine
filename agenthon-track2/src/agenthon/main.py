"""Orchestration only: card -> point-in-time data -> P0 -> text -> P1 -> joint draws -> output.

    try:
        C            = read_card()                              L0
        X, T         = point_in_time_filter(C)                  L1
        grid, state  = transform + build_state                  L2-L4
        M0           = exact official baseline replica          (reference + fail-safe)
        tails        = in-unit residual backtest                L20
        P0           = numerical_forecast(state)                L5-L7
        action       = adaptive policy(context)                 policy (RL, safe improvement)
        evidence     = text_pipeline(T, state)                  L8-L14
        kappa, g     = contradiction, information gate          L15-L16
        P1           = distribution + dependence update         L17-L19
        pool         = joint Monte Carlo(P1); calibrate         L21-L22
        draws        = score-aware decision(pool, M0)           L22b
        write_output(draws)                                     L23
    except:
        fail_safe_forecast()   # exact M0 draws (scores 1.0), never a missed card
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import zlib
from typing import Any

import numpy as np

from .calibration import calibrate
from .card import find_card, parse_card, validate_card
from .card.policy import resolve_policy
from .config import _deep_merge, load_config
from .data import point_in_time_filter
from .decision import decide
from .fusion import (contradiction_score, dependence_update, distribution_update,
                     information_gate, regime_update)
from .numerical import build_grid, build_state, m0_forecast, numerical_forecast
from .output import build_meta, build_rationale, write_forecast
from .policy import choose_action, context_vector
from .resources import vintage_overrides
from .safety import Deadline, check_draws, fail_safe_forecast, run_with_timeout, spread_sanity
from .sampling import sample_belief
from .tails import TailProfile, tail_profile
from .text import text_pipeline
from .types import Card, ForecastResult, PointInTimeData, TextEvidence


def log(msg: str) -> None:
    print(f"[agenthon] {msg}", file=sys.stderr, flush=True)


def card_seed(card: Card, salt: int) -> int:
    env = os.environ.get("QFBENCH_SEED", "0")
    try:
        e = int(env)
    except ValueError:
        e = zlib.crc32(env.encode())
    return (zlib.crc32(card.unit_id.encode()) ^ (salt * 2654435761) ^ (e * 97)) & 0x7FFFFFFF


def forecast_card(card: Card, panels: pathlib.Path, text: pathlib.Path, info_asof: str,
                  cfg: dict[str, Any], client: Any = None) -> ForecastResult:
    pit = point_in_time_filter(card, panels, text, info_asof, cfg)
    return forecast_from_pit(card, pit, info_asof, cfg, client)


def forecast_from_pit(card: Card, pit: PointInTimeData, info_asof: str, cfg: dict[str, Any],
                      client: Any = None) -> ForecastResult:
    """Everything after L1. The offline walk-forward calls this with in-memory history slices."""
    t0 = time.monotonic()
    dl = Deadline(float(cfg["runtime"]["total_seconds"]))
    pol = resolve_policy(card, cfg)
    seed = card_seed(card, int(cfg["runtime"]["seed_salt"]))
    hist = pit.histories
    grid = build_grid(card, hist, int(cfg["m0"]["window"]), float(cfg["m0"]["override_ratio"]))
    m0 = m0_forecast(card, grid, hist, int(cfg["m0"]["draws"]))
    state = build_state(card, grid, hist, cfg)
    prior_df = float(pol.df if pol.df is not None else cfg["numerical"]["df"])
    tails = run_with_timeout(lambda: tail_profile(state, hist, cfg, prior_df),
                             float(cfg["runtime"]["numeric_seconds"]), None) or TailProfile(df=prior_df)
    bias = {a: tails.scale_bias.get(a, 1.0) * pol.scale_mult for a in grid.assets}
    p0 = numerical_forecast(state, hist, cfg, bias, tails.df)
    ledger: list[dict[str, Any]] = [{"step": "numerical_prior", "df": round(tails.df, 2),
                                     "scale_bias": {a: round(b, 3) for a, b in bias.items()}}]
    # numeric-only draws exist from here on: any later failure degrades to these, not to M0
    n_out = int(min(max(pol.n_draws, card.n_draws_min, int(cfg["sampling"]["n_min"])),
                    int(cfg["output"]["max_draws"])))
    path = "full"
    evidence = TextEvidence.empty(grid.assets, "not run")
    x = context_vector(card, state)
    action = choose_action(x, info_asof, cfg)
    try:
        evidence = text_pipeline(pit, state, card, cfg, dl.sub(float(cfg["runtime"]["text_seconds"])),
                                 client)
        kappa, _ = contradiction_score(state, evidence.signal)
        gate = information_gate(evidence, kappa, pol, cfg, action.text_strength)
        state.regime = regime_update(state.regime, evidence.signal, gate)
        p1, led = distribution_update(p0, evidence.signal, gate, tails.abs_z_quantiles, pol, cfg)
        p1, wdep = dependence_update(p1, evidence.signal, gate, pol.dependence_strength)
        ledger += led
        if wdep > 0:
            ledger.append({"step": "dependence", "stress_blend": round(wdep, 4)})
    except Exception as exc:
        log(f"text/fusion degraded to numeric-only: {type(exc).__name__}: {exc}")
        from .types import GateDecision
        gate, p1, path = GateDecision(0.0, 0.0, "none"), p0, "numeric_only"
    classes = {a: hist[a].asset_class for a in grid.assets}
    npool = int(cfg["sampling"]["belief_pool"])
    def pools(belief):
        pl = check_draws(sample_belief(belief, npool, seed, bool(cfg["sampling"]["antithetic"])), grid.d, "pool")
        hd = check_draws(sample_belief(belief, int(cfg["decision"]["eval_outcomes"]), seed + 1), grid.d, "holdout")
        pl, cal = calibrate(pl, grid, classes, info_asof, cfg)
        hd, _ = calibrate(hd, grid, classes, info_asof, cfg)
        return pl, hd, cal

    pool, hold, calib = pools(p1)
    if not spread_sanity(pool, m0.draws) and p1 is not p0:
        # text pushed the spread out of bounds: keep the numerical edge, drop only the text update
        log("text update pushed the spread outside [0.15, 8] x M0; degrading to numeric-only")
        pool, hold, calib = pools(p0)
        path = "numeric_only"
        ledger.append({"step": "text_rejected", "reason": "spread sanity"})
    if not spread_sanity(pool, m0.draws):
        raise RuntimeError("belief spread outside [0.15, 8] x M0 spread: refusing (unit/scale blunder)")
    gamma, lam, note = 0.0, 0.0, ""
    if cfg["decision"].get("enabled", True):
        opt = bool(cfg["decision"].get("optimise", False))
        dec = decide(pool, hold, m0.draws, n_out, card.effective_weights(), card.tail_levels,
                     card.n_cells > 1, cfg, seed + 2,
                     fixed_gamma=None if opt else action.gamma,
                     fixed_lambda=None if opt else action.lam)
        draws, gamma, lam, note = dec.draws, dec.gamma, dec.lam, dec.note
        ledger.append({"step": "score_aware", "gamma": gamma, "lambda": lam, "policy": action.source,
                       "expected_norm_score": {str(k): round(v, 4) for k, v in dec.expected.items()},
                       "ess": round(dec.ess, 1)})
    else:
        draws = pool[:n_out]
    if cfg.get("ablation", {}).get("submit_m0"):
        draws, gamma = m0.draws.copy(), 0.0  # ablation rung M0 (and a sanity check: must score 1.0)
    draws = check_draws(draws, grid.d, "submission")
    sig = evidence.signal
    info = {
        "path": path, "df": tails.df, "gamma": gamma, "lambda": lam, "min_sd_ratio": cfg["decision"]["min_sd_ratio"],
        "calibration": calib, "params": cfg.get("_vintage", "priors"), "ledger": ledger, "note": note,
        "assets": {a: {"class": s.asset_class, "anchor": s.anchor, "last_date": s.last_date,
                       "sd_now": s.step_sd_ewma, "sd_m0": s.step_sd_m0, "sd_long": s.step_sd_long,
                       "model": p0.provenance["assets"].get(a, {}).get("model", "?"),
                       "scale_bias": bias.get(a, 1.0)} for a, s in state.assets.items()},
        "text": {"source": sig.source, "llm_calls": evidence.llm_calls, "n_retrieved": len(evidence.retrieved),
                 "n_events": len(evidence.events), "gate": gate.g, "kappa": gate.kappa,
                 "mechanism": gate.mechanism, "established": sig.established, "summary": sig.summary,
                 "cited": list(sig.cited_docs),
                 "assets": {a: {"p_up": s.p_up, "magnitude": s.magnitude, "vol_view": s.vol_view,
                                "tail_side": s.tail_side} for a, s in sig.assets.items()} if gate.g > 0 else {}},
    }
    return ForecastResult(card=card, grid=grid, draws=draws, path=path, ledger=ledger,
                          diagnostics={**info, "seconds": time.monotonic() - t0, "policy": action.source,
                                       "m0_draws": m0.draws, "pool": pool, "context": x,
                                       "expected": dec.expected if cfg["decision"].get("enabled", True) else {}},
                          seed=seed)


def run(panels: pathlib.Path, text: pathlib.Path, asof: str | None, out: pathlib.Path,
        card_path: pathlib.Path | None = None, profile: str | None = None,
        overrides: dict[str, Any] | None = None, client: Any = None) -> tuple[str, ForecastResult | None]:
    card: Card | None = None
    try:
        card = parse_card(find_card(panels, card_path))
        info_asof = validate_card(card, asof)
        cfg = load_config(profile, card.family, overrides)
        vo, label = vintage_overrides(info_asof, card.family)
        if vo:
            cfg = _deep_merge(cfg, vo)
            if overrides:
                cfg = _deep_merge(cfg, overrides)  # explicit overrides always win
        cfg["_vintage"] = label
        res = forecast_card(card, panels, text, info_asof, cfg, client)
        meta = build_meta(card, res.draws.shape[0], f"agenthon {res.path}", res.seed)
        write_forecast(out, res.draws, res.grid, card, meta,
                       build_rationale(card, res.grid, res.draws, res.diagnostics))
        log(f"{card.unit_id}: path={res.path} gamma={res.diagnostics.get('gamma')} "
            f"g={res.diagnostics['text']['gate']:.3f} n={res.draws.shape[0]} "
            f"{res.diagnostics['seconds']:.1f}s")
        return res.path, res
    except BaseException as exc:  # noqa: BLE001 - includes KeyboardInterrupt/SystemExit from deep code
        if isinstance(exc, SystemExit) and card is None:
            raise
        return fail_safe_forecast(card, panels, out, exc), None
