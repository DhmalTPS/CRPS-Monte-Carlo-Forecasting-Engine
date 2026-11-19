"""L21 — joint Monte Carlo scenario generator: JointBelief -> draws [n, d].

Construction (guarantees a coherent joint scenario per draw index):
  1. scenario j for draw m (stratified: exact counts proportional to the mixture weights),
  2. one variance-preserving t-mixing scale per draw (shared by all cells: tail dependence),
  3. correlated Gaussian increments between consecutive step points, increment variance
     V_a(p_k) - V_a(p_{k-1}) >= 0, cumulated into a path -> cross-horizon covariance is exactly
     the path structure the variogram term rewards (the future at 63 BD contains the one at 21),
  4. value = mu + shift_j + scale_j * path, then the soft floor where one applies.
Antithetic pairs (z, -z) halve Monte-Carlo noise in the centre at no cost.
"""

from __future__ import annotations

import numpy as np

from ..types import JointBelief
from .copula import soft_floor, t_mixing
from .dependence import safe_cholesky


def _stratified_labels(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    counts = np.floor(w * n).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        frac = w * n - counts
        extra = rng.choice(len(w), size=rem, replace=False if rem <= len(w) else True,
                           p=frac / frac.sum() if frac.sum() > 0 else None)
        for e in np.atleast_1d(extra):
            counts[e] += 1
    labels = np.repeat(np.arange(len(w)), counts)
    rng.shuffle(labels)
    return labels


def sample_paths(belief: JointBelief, n: int, rng: np.random.Generator,
                 antithetic: bool = True) -> np.ndarray:
    """Standardised path increments cumulated per asset: [n, n_assets, K] (unit-free Gaussian*mix)."""
    assets = belief.grid.assets
    K = len(belief.step_points)
    L = safe_cholesky(belief.corr)
    half = (n + 1) // 2 if antithetic else n
    z = rng.standard_normal((half, K, len(assets))) @ L.T  # [half, K, A]
    if antithetic:
        z = np.concatenate([z, -z], axis=0)[:n]
    inc_sd = np.empty((K, len(assets)))
    for ai, a in enumerate(assets):
        vp = np.maximum.accumulate(np.maximum(belief.var_path[a], 0.0))
        dv = np.diff(np.concatenate([[0.0], vp]))
        inc_sd[:, ai] = np.sqrt(np.maximum(dv, 0.0))
    path = np.cumsum(z * inc_sd[None, :, :], axis=1)  # [n, K, A]
    return path


def sample_belief(belief: JointBelief, n: int, seed: int, antithetic: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = belief.grid
    path = sample_paths(belief, n, rng, antithetic)
    mix = t_mixing(rng, n, belief.df)
    labels = _stratified_labels(rng, n, np.array([s.weight for s in belief.scenarios]))
    aidx = {a: i for i, a in enumerate(grid.assets)}
    kidx = {p: k for k, p in enumerate(belief.step_points)}
    base = np.empty((n, grid.d))
    for i, c in enumerate(grid.cells):
        base[:, i] = path[:, kidx[c.steps], aidx[c.asset]] * mix
    shift = np.stack([s.shift for s in belief.scenarios])  # [J, d]
    scale = np.stack([s.scale for s in belief.scenarios])  # [J, d]
    out = belief.mu[None, :] + shift[labels] + scale[labels] * base
    for a, fl in belief.floors.items():
        for i in grid.cells_of(a):
            out[:, i] = soft_floor(out[:, i], fl)
    return out
