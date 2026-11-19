"""L5 Tier-1 — the statistical forecasting core, written as pure vectorised functions.

The same functions serve three callers, which is what keeps research and production identical:
the live forecast (evaluated at the last index), the in-unit self-calibration backtest (evaluated
at past cutoffs of the unit's own history), and the offline walk-forward lab.

Model (per asset, steps x_t = level differences or log returns)
---------------------------------------------------------------
    sigma2_now(t) = EWMA_H(x^2)(t)                              current conditional variance
    sigma2_lr(t)  = (1 - omega) * Var_300(x)(t) + omega * Var_L(x)(t)   long-run anchor
    V(s | t)      = c^2 * VR(s) * [ s*sigma2_lr + (sigma2_now - sigma2_lr) * (1 - phi^s)/(1 - phi) ]

the textbook GARCH(1,1) variance term structure (conditional variance decays geometrically toward
its long-run level with persistence phi), multiplied by a shrunk empirical variance ratio VR(s)
that absorbs serial dependence of steps (momentum or mean reversion at the horizon).

    centre(s | t) = anchor + s * [ kappa * mu_300(t) + kappa_lr * mu_L(t) ]

M0 uses kappa = 1 on the trailing-300 mean. For near-martingale series (rates, FX) the trailing
mean is almost pure estimation noise, so kappa is tuned in walk-forward (prior: 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_var_path(x: np.ndarray, halflife: float, warmup: int = 20) -> np.ndarray:
    """EWMA of squared steps (uncentred: the drift of daily steps is negligible vs their scale)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    sq = pd.Series(x * x)
    seed = float(np.mean(sq.iloc[:warmup])) if len(sq) else 0.0
    v = sq.ewm(halflife=halflife, adjust=False).mean().to_numpy()
    # blend in the warm-up so the first values are not a single squared step
    n = np.arange(1, len(x) + 1)
    wt = np.clip(n / warmup, 0.0, 1.0)
    return wt * v + (1 - wt) * seed


def rolling_var(x: np.ndarray, n: int, min_periods: int = 20) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=min(min_periods, n)).var(ddof=1).to_numpy()


def rolling_mean(x: np.ndarray, n: int, min_periods: int = 20) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=min(min_periods, n)).mean().to_numpy()


def aggregate_variance(now: np.ndarray | float, lr: np.ndarray | float, s: float,
                       phi: float) -> np.ndarray | float:
    """Sum of s one-step conditional variances under geometric mean reversion."""
    phi = float(np.clip(phi, 0.0, 0.99999))
    if phi <= 0.0:
        geo = 1.0
    else:
        geo = (1.0 - phi ** s) / (1.0 - phi)
    out = s * np.asarray(lr) + (np.asarray(now) - np.asarray(lr)) * geo
    return np.maximum(out, 1e-18 + 0.05 * s * np.asarray(lr))


def variance_ratio(x: np.ndarray, s: int, n0: float = 40.0,
                   clip: tuple[float, float] = (0.6, 1.6), max_len: int = 5000) -> float:
    """Shrunk empirical variance ratio Var(sum of s steps) / (s Var(step)) from overlapping sums.

    n_eff = len/s independent blocks; shrink toward 1 with pseudo-count n0 (empirical Bayes).
    """
    x = np.asarray(x, dtype=float)[-max_len:]
    x = x[np.isfinite(x)]
    if s <= 1 or len(x) < 3 * s + 20:
        return 1.0
    v1 = float(np.var(x, ddof=1))
    if v1 <= 0:
        return 1.0
    cs = np.concatenate([[0.0], np.cumsum(x - x.mean())])
    agg = cs[s:] - cs[:-s]
    vr = float(np.var(agg, ddof=1)) / (s * v1)
    n_eff = len(x) / s
    w = n_eff / (n_eff + n0)
    return float(np.clip(1.0 + w * (vr - 1.0), clip[0], clip[1]))


def vol_term_structure(sig_now: float, sig_lr: float, steps: list[int], phi: float,
                       vr: dict[int, float] | None = None, scale: float = 1.0) -> np.ndarray:
    """Cumulative variance at each step count (monotone non-decreasing by construction)."""
    out = []
    for s in steps:
        v = float(aggregate_variance(sig_now ** 2, sig_lr ** 2, float(s), phi))
        v *= (vr or {}).get(int(s), 1.0)
        out.append(v * scale * scale)
    arr = np.maximum.accumulate(np.asarray(out, dtype=float))
    return arr


# --------------------------------------------------------------------------------------------
# Monthly macro (CPI/PCE indices, payrolls, unemployment)
# --------------------------------------------------------------------------------------------
def macro_drift_and_sd(dx: np.ndarray, halflife: float = 6.0, trend_weight: float = 0.7,
                       window: int = 300) -> tuple[float, float, float]:
    """(drift per month, innovation sd, drift-estimate sd) for monthly changes.

    Inflation and payroll growth are strongly persistent, so the recent trend (EWMA of changes) is
    a far better drift than a 25-year mean. Innovation sd is measured around the EWMA trend on a
    robust (MAD) scale so a single COVID-type print does not dominate; the drift uncertainty term
    turns s-month variance into s*sd^2 + s^2*sd_mu^2 (a slowly moving trend).
    """
    dx = np.asarray(dx, dtype=float)[-window:]
    dx = dx[np.isfinite(dx)]
    if len(dx) < 6:
        m = float(np.mean(dx)) if len(dx) else 0.0
        sd = float(np.std(dx)) if len(dx) > 1 else 0.0
        return m, sd, sd
    tr = pd.Series(dx).ewm(halflife=halflife, adjust=False).mean().to_numpy()
    drift = trend_weight * tr[-1] + (1 - trend_weight) * float(np.mean(dx))
    resid = dx[1:] - tr[:-1]
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    sd = float(np.sqrt(0.5 * mad ** 2 + 0.5 * np.var(resid[-60:], ddof=1))) if len(resid) > 3 else mad
    sd_mu = float(np.std(np.diff(tr[-60:]), ddof=1)) * np.sqrt(halflife) if len(tr) > 10 else sd
    return float(drift), max(sd, 1e-12), max(sd_mu, 0.0)


# --------------------------------------------------------------------------------------------
# Transfer (F2): target seen only as an early window plus the as-of level
# --------------------------------------------------------------------------------------------
def transfer_relative_vol(early_steps_rel: np.ndarray, proxy_rel_vols: list[float],
                          proxy_weight: float = 0.5, em_premium: float = 1.25) -> float:
    """Daily relative (log) vol for an unseen-path currency.

    The early window says what kind of series it is (a managed rate has tiny early vol); the G10
    cross-section says what currency vol looks like *now*. Neither alone is trustworthy: blend in
    log-variance space, with an EM premium on the proxy. The text layer can later move the regime
    (managed vs stressed) through the scenario mixture.
    """
    e = np.asarray(early_steps_rel, dtype=float)
    e = e[np.isfinite(e)]
    early = float(np.std(e, ddof=1)) if len(e) > 20 else np.nan
    prox = float(np.median(proxy_rel_vols)) * em_premium if proxy_rel_vols else np.nan
    if not np.isfinite(early) and not np.isfinite(prox):
        return 0.005
    if not np.isfinite(early):
        return prox
    if not np.isfinite(prox):
        return early
    lv = (1 - proxy_weight) * np.log(max(early, 1e-7) ** 2) + proxy_weight * np.log(prox ** 2)
    return float(np.sqrt(np.exp(lv)))
