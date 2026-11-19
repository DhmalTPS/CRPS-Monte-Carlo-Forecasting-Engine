"""NaN/inf rows in a panel are dropped at load; the forecast stays finite."""

import numpy as np
import pandas as pd

from agenthon.main import run


def test_nan_and_inf_rows(unit, out_dir):
    p = unit / "panels" / "rates_daily.parquet"
    df = pd.read_parquet(p)
    df.loc[df.sample(frac=0.05, random_state=0).index, "value"] = np.nan
    df.loc[df.sample(frac=0.01, random_state=1).index, "value"] = np.inf
    df.to_parquet(p, index=False)
    path, _ = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                  overrides={"llm": {"enabled": False}})
    out = pd.read_parquet(out_dir / "forecast.parquet")
    assert path == "full" and np.isfinite(out.value).all()
