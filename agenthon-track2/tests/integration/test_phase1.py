"""Phase 1: ANY valid card -> valid forecast.parquet + forecast_meta.json + forecast_rationale.md."""

import json

import pandas as pd

from agenthon.main import run


def test_full_path_numeric(unit, out_dir):
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                    overrides={"llm": {"enabled": False}})
    assert path == "full" and res is not None
    assert sorted(p.name for p in out_dir.iterdir()) == ["forecast.parquet", "forecast_meta.json",
                                                         "forecast_rationale.md"]
    df = pd.read_parquet(out_dir / "forecast.parquet")
    meta = json.loads((out_dir / "forecast_meta.json").read_text())
    assert meta["n_draws"] == df.draw.nunique() >= 500
    assert set(df.asset) == {"UST_2Y", "UST_10Y"} and set(df.horizon) == {21, 63}
    assert (out_dir / "forecast_rationale.md").read_text(encoding="utf-8").strip()


def test_single_cell_card(single_unit, out_dir):
    path, res = run(single_unit / "panels", single_unit / "text", None, out_dir / "forecast.parquet",
                    overrides={"llm": {"enabled": False}})
    assert path == "full"
    assert pd.read_parquet(out_dir / "forecast.parquet").draw.nunique() >= 1000  # F4 draw floor


def test_deterministic_given_seed(unit, tmp_path):
    outs = []
    for k in range(2):
        od = tmp_path / f"o{k}"
        od.mkdir()
        run(unit / "panels", unit / "text", None, od / "forecast.parquet", overrides={"llm": {"enabled": False}})
        outs.append(pd.read_parquet(od / "forecast.parquet"))
    assert outs[0].equals(outs[1])
