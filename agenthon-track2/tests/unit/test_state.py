import numpy as np

from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.numerical import build_grid, build_state
from agenthon.numerical.statistical import aggregate_variance, variance_ratio, vol_term_structure


def _state(unit):
    c = parse_card(unit / "card.toml")
    cfg = load_config()
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, cfg)
    g = build_grid(c, pit.histories)
    return build_state(c, g, pit.histories, cfg), g


def test_grid_in_m0_order(unit):
    _, g = _state(unit)
    assert [(c.asset, c.horizon) for c in g.cells] == [("UST_10Y", 21), ("UST_10Y", 63),
                                                       ("UST_2Y", 21), ("UST_2Y", 63)]


def test_state_is_finite_and_correlation_valid(unit):
    st, _ = _state(unit)
    for s in st.assets.values():
        assert s.step_sd_ewma > 0 and s.step_sd_long > 0 and np.isfinite(s.momentum_z)
    w = np.linalg.eigvalsh(st.step_corr)
    assert np.all(w > 0) and np.allclose(np.diag(st.step_corr), 1.0)
    assert abs(st.regime.probs.sum() - 1) < 1e-12


def test_term_structure_limits():
    # phi = 0 -> no persistence: V(s) = s*lr + (now - lr)
    assert np.isclose(aggregate_variance(4.0, 1.0, 10, 0.0), 10 + 3)
    v = vol_term_structure(2.0, 1.0, [1, 21, 63], 0.99)
    assert np.all(np.diff(v) >= 0) and v[0] > 0


def test_variance_ratio_shrinks_to_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    assert abs(variance_ratio(x, 21) - 1.0) < 0.2
    assert variance_ratio(x[:50], 21) == 1.0
