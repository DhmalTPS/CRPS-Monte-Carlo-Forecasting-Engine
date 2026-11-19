"""Output contract + metric parity with the official implementation."""

import json

import numpy as np
import pandas as pd
import pytest

from agenthon.card import parse_card
from agenthon.decision import m0_scoring as ms
from agenthon.errors import OutputValidationError
from agenthon.output import build_meta, write_forecast
from agenthon.types import Cell, Grid


def _grid():
    return Grid(tuple(Cell(a, h, h) for a in ("UST_10Y", "UST_2Y") for h in (21, 63)))


def test_writer_emits_exactly_three_files_with_dtypes(unit, out_dir):
    c = parse_card(unit / "card.toml")
    d = np.random.default_rng(0).standard_normal((300, 4))
    write_forecast(out_dir / "forecast.parquet", d, _grid(), c, build_meta(c, 300, "t", 0), "# r\n")
    assert sorted(p.name for p in out_dir.iterdir()) == ["forecast.parquet", "forecast_meta.json",
                                                         "forecast_rationale.md"]
    df = pd.read_parquet(out_dir / "forecast.parquet")
    assert str(df.draw.dtype) == "int32" and str(df.horizon.dtype) == "int32"
    assert str(df.value.dtype) == "float64" and len(df) == 1200
    assert df.groupby("draw").size().eq(4).all()
    meta = json.loads((out_dir / "forecast_meta.json").read_text())
    assert meta["unit_id"] == c.unit_id and meta["asof"] == c.asof and meta["n_draws"] == 300


@pytest.mark.parametrize("bad", ["nan", "few", "flat"])
def test_writer_refuses_inadmissible(unit, out_dir, bad):
    c = parse_card(unit / "card.toml")
    d = np.random.default_rng(0).standard_normal((300, 4))
    if bad == "nan":
        d[0, 0] = np.nan
    if bad == "few":
        d = d[:100]
    if bad == "flat":
        d[:, 1] = 1.0
    with pytest.raises(OutputValidationError):
        write_forecast(out_dir / "forecast.parquet", d, _grid(), c, build_meta(c, len(d), "t", 0), "# r")


def test_metric_parity_with_official():
    off = pytest.importorskip("qfbench2_common.scoring.crps")
    tail = pytest.importorskip("qfbench2_track_forecasting.tail")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 5)) * [1, 2, 3, 4, 5]
    y = rng.standard_normal(5)
    assert abs(ms.crps_marginal(x, y[None])[0] - off.crps_marginal(x, y)) < 1e-12
    assert abs(ms.variogram(x, y[None])[0] - off.variogram_score(x, y, p=0.5)) < 1e-9
    assert abs(ms.pinball(x, y[None])[0] - tail.tail_pinball(x, y, (0.01, 0.05, 0.95, 0.99))) < 1e-12


def test_m0_scores_exactly_one():
    rng = np.random.default_rng(2)
    m0 = rng.standard_normal((500, 3))
    y = rng.standard_normal((50, 3))
    c0 = ms.components(m0, y)
    assert np.allclose(ms.normalised(c0, c0, (0.5, 0.3, 0.2)), 1.0)
