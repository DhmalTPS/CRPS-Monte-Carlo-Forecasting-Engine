"""Empirical-Bayes shrinkage for thin samples (eq. 23 of the design document).

theta_post = omega * theta_local + (1 - omega) * theta_global,  omega = n_eff / (n_eff + n0).
Overlapping s-step windows at stride k carry roughly n * k / s independent observations; that
n_eff (not the raw count) is what enters omega, otherwise overlap would fake precision.
"""

from __future__ import annotations

import numpy as np


def n_effective(n: int, stride: int, s: int) -> float:
    return float(max(1.0, n * min(1.0, stride / max(s, 1))))


def shrink_log_scale(z: np.ndarray, stride: int, s: int, n0: float,
                     clip: tuple[float, float]) -> tuple[float, float]:
    """(posterior multiplicative scale bias, weight on the local estimate)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 20:
        return 1.0, 0.0
    zc = np.clip(z, -8, 8)
    # robust second moment: average of RMS and MAD-based sd (both consistent under normality)
    rms = float(np.sqrt(np.mean(zc ** 2)))
    mad = float(np.median(np.abs(zc - np.median(zc)))) * 1.4826
    local = np.sqrt(max(rms, 1e-6) * max(mad, 1e-6))
    ne = n_effective(len(z), stride, s)
    w = ne / (ne + n0)
    post = float(np.exp(w * np.log(local)))
    return float(np.clip(post, clip[0], clip[1])), float(w)


def shrink_df(z: np.ndarray, stride: int, s: int, prior_df: float, n0: float,
              clip: tuple[float, float]) -> float:
    """Tail thickness via excess kurtosis -> Student-t dof (nu = 4 + 6/k), shrunk in 1/nu."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 40:
        return prior_df
    zc = np.clip(z / max(np.std(z), 1e-9), -10, 10)
    k = float(np.mean(zc ** 4) - 3.0)
    local = 4.0 + 6.0 / k if k > 0.05 else 200.0
    ne = n_effective(len(z), stride, s)
    w = ne / (ne + n0)
    inv = w / local + (1 - w) / prior_df
    return float(np.clip(1.0 / max(inv, 1e-6), clip[0], clip[1]))
