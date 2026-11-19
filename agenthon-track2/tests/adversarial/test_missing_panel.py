"""A missing target series / unreadable panel must still produce an admissible forecast."""

import pandas as pd

from agenthon.main import run


def test_corrupt_context_panel_is_skipped(unit, out_dir):
    (unit / "panels" / "zz_corrupt.parquet").write_bytes(b"not a parquet")
    path, _ = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                  overrides={"llm": {"enabled": False}})
    assert path == "full"


def test_target_absent_goes_failsafe_constant(unit, out_dir):
    p = unit / "panels" / "rates_daily.parquet"
    df = pd.read_parquet(p)
    df[df.asset != "UST_2Y"].to_parquet(p, index=False)
    path, _ = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                  overrides={"llm": {"enabled": False}})
    assert path.startswith("failsafe")
    assert sorted(x.name for x in out_dir.iterdir()) == ["forecast.parquet", "forecast_meta.json",
                                                         "forecast_rationale.md"]
