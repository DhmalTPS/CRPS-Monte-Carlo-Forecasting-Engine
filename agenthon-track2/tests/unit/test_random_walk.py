"""The M0 replica must follow docs/M0-BASELINE.md to the letter."""

import zlib

import numpy as np
import pytest

from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.numerical import build_grid, m0_forecast, m0_seed

import pathlib

REF_UNITS = pathlib.Path(__file__).resolve().parents[3] / "_reference" / "track2-forecasting-public" / "units"


def test_seed_rule():
    assert m0_seed("t2-EXAMPLE-ust-curve-1m") == 795546941
    assert m0_seed("x") == zlib.crc32(b"x") & 0x7FFFFFFF


def test_m0_path_covariance(unit):
    c = parse_card(unit / "card.toml")
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, load_config())
    g = build_grid(c, pit.histories)
    f = m0_forecast(c, g, pit.histories)
    assert f.draws.shape == (500, 4)
    i21, i63 = g.index_of("UST_2Y", 21), g.index_of("UST_2Y", 63)
    assert np.isclose(f.cov[i21, i63], f.cov[i21, i21])  # min(s_i, s_j) * Sigma
    assert np.isclose(f.cov[i63, i63] / f.cov[i21, i21], 3.0)
    # deterministic
    assert np.array_equal(f.draws, m0_forecast(c, g, pit.histories).draws)


@pytest.mark.skipif(not REF_UNITS.is_dir(), reason="reference units not checked out")
def test_m0_worked_example_numbers():
    u = REF_UNITS / "t2-EXAMPLE-ust-curve-1m"
    c = parse_card(u / "card.toml")
    pit = point_in_time_filter(c, u / "panels", u / "text", c.asof, load_config())
    g = build_grid(c, pit.histories)
    assert all(cell.steps == 21 for cell in g.cells)  # counted 24 vs 21 -> declared kept
    assert m0_forecast(c, g, pit.histories).n_rows == 128


@pytest.mark.skipif(not REF_UNITS.is_dir(), reason="reference units not checked out")
def test_m0_monthly_step_counts():
    u = REF_UNITS / "t2-F1-cpi-glidepath-2023"
    c = parse_card(u / "card.toml")
    pit = point_in_time_filter(c, u / "panels", u / "text", c.asof, load_config())
    assert [cell.steps for cell in build_grid(c, pit.histories).cells] == [8, 9]
