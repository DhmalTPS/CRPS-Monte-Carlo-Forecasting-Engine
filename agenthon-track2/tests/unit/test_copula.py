"""L19/L21: joint draws preserve correlation and path structure; t-mixing preserves variance."""

import numpy as np

from agenthon.sampling.copula import soft_floor, t_mixing
from agenthon.sampling.dependence import blend_correlation, stress_correlation
from agenthon.sampling import sample_belief
from agenthon.types import Cell, Grid, JointBelief, Scenario


def test_path_and_cross_asset_structure():
    g = Grid((Cell("A", 21, 21), Cell("A", 63, 63), Cell("B", 21, 21), Cell("B", 63, 63)))
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    b = JointBelief(g, np.zeros(4), (21, 63), {"A": np.array([21.0, 63.0]), "B": np.array([21.0, 63.0])},
                    corr, 1000.0, [Scenario("base", 1.0, np.zeros(4), np.ones(4))])
    d = sample_belief(b, 40000, 3)
    c = np.corrcoef(d, rowvar=False)
    assert abs(c[0, 2] - 0.8) < 0.03  # cross-asset at the same horizon
    assert abs(c[0, 1] - np.sqrt(21 / 63)) < 0.03  # cross-horizon = path structure
    assert abs(d[:, 1].var() / d[:, 0].var() - 3.0) < 0.15


def test_t_mixing_variance_preserving():
    rng = np.random.default_rng(0)
    w = t_mixing(rng, 400000, 7.0)
    assert abs(np.mean(w ** 2) - 1.0) < 0.02


def test_correlation_updates_stay_psd():
    c = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    s = stress_correlation(c, 0.6)
    for g in (0.0, 0.3, 1.0):
        m = blend_correlation(c, s, g)
        assert np.all(np.linalg.eigvalsh(m) > -1e-10) and np.allclose(np.diag(m), 1.0)


def test_soft_floor_monotone_and_identity_above():
    y = np.linspace(-2, 3, 200)
    f = soft_floor(y, -0.1)
    assert np.all(np.diff(f) >= 0) and np.allclose(f[y > 0.2], y[y > 0.2]) and f.min() > -1.5
