"""Composite and baseline-normalised score, plus parity against the official implementation.

    S_norm(card) = min(4, sum_k w_k * S_k(forecast, y) / S_k(M0, y))

with single-cell weights renormalised to (0.714, 0, 0.286). delta_text = S(P0) - S(P1) > 0 means
the text helped on that card (eq. 9).
"""

from __future__ import annotations

import numpy as np

from agenthon.decision import m0_scoring as ms


def normalised_score(draws: np.ndarray, m0_draws: np.ndarray, y: np.ndarray, weights,
                     levels=(0.01, 0.05, 0.95, 0.99)) -> float:
    y = np.atleast_2d(y)
    joint = y.shape[1] > 1
    c = ms.components(draws, y, levels, joint)
    c0 = ms.components(m0_draws, y, levels, joint)
    return float(ms.normalised(c, c0, weights)[0])


def delta_text(draws_p0: np.ndarray, draws_p1: np.ndarray, m0_draws: np.ndarray, y, weights) -> float:
    return normalised_score(draws_p0, m0_draws, y, weights) - normalised_score(draws_p1, m0_draws, y, weights)


def parity_check(seed: int = 0) -> dict[str, float]:
    """Max abs difference vs qfbench2_common.scoring.crps and the track tail module."""
    from qfbench2_common.scoring import crps as official
    from qfbench2_track_forecasting.tail import tail_pinball

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((500, 6)) * [1, 2, 3, 1, 1, 5]
    y = rng.standard_normal(6)
    return {
        "crps": abs(ms.crps_marginal(x, y[None])[0] - official.crps_marginal(x, y)),
        "variogram": abs(ms.variogram(x, y[None])[0] - official.variogram_score(x, y, p=0.5)),
        "pinball": abs(ms.pinball(x, y[None])[0] - tail_pinball(x, y, (0.01, 0.05, 0.95, 0.99))),
    }
