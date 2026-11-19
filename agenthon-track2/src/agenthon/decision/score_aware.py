"""Score-aware decision layer (L22b): choose the submitted draws to minimise the ACTUAL loss.

The leaderboard averages, per card, S(Q, y) / S(M0, y) component-wise. For a proper score S and a
positive outcome weight w(y), argmin_Q E_P[w(y) S(Q, y)] is the reweighted belief
Q* ∝ w(y) p(y) (because E_P[w S(Q,.)] = Z * E_{P_w}[S(Q,.)]). With w = 1/S_M0 the metric-optimal
submission is therefore our honest belief P *importance-reweighted toward outcomes where the
baseline scores well*. We use the tempered family w^gamma, gamma in [0, 1]: gamma = 0 submits the
honest belief; the composite mixes three components with different denominators so the optimum is
interior (simulation and walk-forward both put it near 0.3-0.6).

Safeguards (the point is a better expected score, never a laundered point forecast):
* outcomes used to *evaluate* a candidate are drawn from a ROBUST belief (1 - r) P + r M0, not
  from P alone, and are disjoint from the pool Q is resampled from (no self-confirmation);
* no candidate may shrink any cell's sd below `min_sd_ratio` x the honest belief's sd;
* gamma is picked from a small fixed grid (no continuous search to overfit on);
* the decision can only fall back to gamma = 0 on any numerical problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import m0_scoring as ms


@dataclass
class Decision:
    draws: np.ndarray
    gamma: float
    expected: dict[str, float]
    ess: float
    note: str = ""
    lam: float = 0.0


def _m0_weight(pool: np.ndarray, m0: np.ndarray, weights, levels, joint: bool) -> np.ndarray:
    comp0 = ms.components(m0, pool, levels, joint)  # [K, 3]
    w = np.asarray(weights, dtype=float)
    scale = np.where(comp0.mean(axis=0) > 0, comp0.mean(axis=0), 1.0)
    proxy = (comp0 / scale * w).sum(axis=1)  # M0's normalised-proxy loss at each outcome
    return 1.0 / np.maximum(proxy, 1e-12)


def resample(pool: np.ndarray, base_w: np.ndarray, gamma: float, n: int,
             rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if gamma <= 0:
        idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
        return pool[idx], float(len(pool))
    lw = gamma * np.log(base_w)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    ess = float(1.0 / np.sum(p * p))
    # systematic resampling: low variance, preserves the weighted law
    u = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(p), u)
    idx = np.clip(idx, 0, len(pool) - 1)
    return pool[idx], ess


def mix_m0(q: np.ndarray, m0: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction lam of the draws by draws of the baseline's own law (Bayesian model
    averaging with M0). CRPS is convex in the forecast, so ratio(mix) <= (1-lam) ratio(q) + lam:
    this caps the blow-up cards of a ratio metric. Shuffled so draw ids carry no structure."""
    k = int(round(float(lam) * len(q)))
    if k <= 0:
        return q
    out = np.vstack([q[: len(q) - k], m0[rng.integers(0, len(m0), k)]])
    return out[rng.permutation(len(out))]


def decide(pool: np.ndarray, holdout: np.ndarray, m0: np.ndarray, n: int, card_weights,
           levels, joint: bool, cfg: dict[str, Any], seed: int,
           gamma_grid: list[float] | None = None, fixed_gamma: float | None = None,
           fixed_lambda: float | None = None) -> Decision:
    dcfg = cfg["decision"]
    rng = np.random.default_rng(seed)
    honest_sd = pool.std(axis=0)
    base_w = _m0_weight(pool, m0, card_weights, levels, joint)
    min_ratio = float(dcfg.get("min_sd_ratio", 0.6))

    def candidate(g: float, lam: float):
        q, ess = resample(pool, base_w, g, n, rng)
        if g > 0 and (np.any(q.std(axis=0) < min_ratio * honest_sd) or ess < 0.1 * len(pool)):
            return None
        q = mix_m0(q, m0, lam, rng)
        return (q, ess) if np.all(np.isfinite(q)) else None

    if fixed_gamma is not None:
        lam = float(fixed_lambda if fixed_lambda is not None else dcfg.get("lambda", 0.0))
        g = float(fixed_gamma)
        while g > 0:  # step gamma down until the honesty floor is respected
            c = candidate(g, lam)
            if c is not None:
                return Decision(c[0], g, {}, c[1], "fixed action", lam)
            g = round(g - 0.25, 4)
        c = candidate(0.0, lam)
        return Decision(c[0], 0.0, {}, c[1], "fixed action stepped down to gamma 0", lam)
    grid = list(gamma_grid if gamma_grid is not None else dcfg.get("gamma_grid", [0.0, 0.5, 1.0]))
    lams = list(dcfg.get("lambda_grid", [0.0, 0.2]))
    if 0.0 not in grid:
        grid = [0.0] + grid
    r = float(dcfg.get("robust_mix", 0.3))
    k_out = min(int(dcfg.get("eval_outcomes", 3000)), len(holdout))
    n_m0 = int(round(r * k_out))
    outc = np.vstack([holdout[:k_out - n_m0],
                      m0[rng.integers(0, len(m0), size=n_m0)] if n_m0 else holdout[:0]])
    comp_m0 = ms.components(m0, outc, levels, joint)
    expected: dict[str, float] = {}
    best = (np.inf, 0.0, 0.0, None, float(len(pool)))
    for g in grid:
        for lam in lams:
            c = candidate(g, lam)
            if c is None:
                continue
            val = float(ms.normalised(ms.components(c[0], outc, levels, joint), comp_m0,
                                      card_weights).mean())
            expected[f"g{g}_l{lam}"] = val
            if val < best[0] - 1e-9:
                best = (val, g, lam, c[0], c[1])
    if best[3] is None:
        q, ess = resample(pool, base_w, 0.0, n, rng)
        return Decision(q, 0.0, expected, ess, "fallback: no admissible candidate", 0.0)
    return Decision(best[3], best[1], expected, best[4], "", best[2])
