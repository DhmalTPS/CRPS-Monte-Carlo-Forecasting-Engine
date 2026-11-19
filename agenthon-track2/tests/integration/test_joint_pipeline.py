"""Joint structure survives the whole pipeline (score-aware resampling keeps joint draw rows)."""

import numpy as np
import pandas as pd

from agenthon.main import run


def test_joint_rows_coherent(unit, out_dir):
    run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet", overrides={"llm": {"enabled": False}})
    df = pd.read_parquet(out_dir / "forecast.parquet")
    wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    c = np.corrcoef(wide.to_numpy(), rowvar=False)
    # the synthetic panel has strongly correlated tenors and path structure across horizons
    assert c[wide.columns.get_loc(("UST_2Y", 21)), wide.columns.get_loc(("UST_10Y", 21))] > 0.3
    assert c[wide.columns.get_loc(("UST_2Y", 21)), wide.columns.get_loc(("UST_2Y", 63))] > 0.3
