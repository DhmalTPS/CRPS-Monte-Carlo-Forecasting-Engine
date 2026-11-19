import numpy as np

from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.numerical import build_grid, build_state, m0_forecast, numerical_forecast
from agenthon.sampling import sample_belief


def test_p0_is_valid_and_near_m0_scale(unit):
    c = parse_card(unit / "card.toml")
    cfg = load_config()
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, cfg)
    g = build_grid(c, pit.histories)
    st = build_state(c, g, pit.histories, cfg)
    p0 = numerical_forecast(st, pit.histories, cfg)
    assert p0.mu.shape == (g.d,) and len(p0.scenarios) == 1
    for a in g.assets:
        assert np.all(np.diff(p0.var_path[a]) >= 0)
    d = sample_belief(p0, 4000, 1)
    m0 = m0_forecast(c, g, pit.histories)
    ratio = d.std(0) / m0.draws.std(0)
    assert np.all((ratio > 0.4) & (ratio < 2.5))
    # driftless level centre = anchor
    anchors = np.array([st.assets[cell.asset].anchor for cell in g.cells])
    assert np.allclose(p0.mu, anchors)
