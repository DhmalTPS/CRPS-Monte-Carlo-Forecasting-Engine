"""Shared fixtures: self-contained synthetic units (no dependency on the reference checkout)."""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_UNITS = ROOT.parent / "_reference" / "track2-forecasting-public" / "units"

CARD = """
schema_version = "2.0"
[task]
id = "{uid}"
track = "forecasting"
title = "Synthetic test card"
[metadata]
category = "{fam}"
description = "Synthetic unit for tests. Fed tightening narrative."
target_type = "{tt}"
target_frequency = "daily"
[provenance]
data_cutoff = "{asof}"
[scoring.params]
tail_levels = [0.01, 0.05, 0.95, 0.99]
joint = "variogram"
representation = "samples"
n_draws_min = 200
[scoring.params.weights]
marginal = 0.5
joint = 0.3
tail = 0.2
[text]
cutoff = "{asof}"
notes = "Hawkish statement."
[targets]
asset_ids = {assets}
horizons = {horizons}
target_type = "{tt}"
target_frequency = "daily"
value_unit = "percent_per_annum"
"""


def make_panel(assets: list[str], asof: str, n: int = 1500, kind: str = "rates", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=asof, periods=n)
    rows = []
    common = rng.standard_normal(n)
    for i, a in enumerate(assets):
        if kind == "factor":
            v = 0.0003 + 0.01 * (0.7 * common + 0.7 * rng.standard_normal(n))
        else:
            steps = 0.05 * (0.8 * common + 0.6 * rng.standard_normal(n))
            v = 3.0 + i * 0.3 + np.cumsum(steps)
        rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "asset": a, "value": v,
                                  "panel_id": f"{kind}_daily"}))
    return pd.concat(rows, ignore_index=True)


def write_unit(root: pathlib.Path, uid: str = "t2-F3-synthetic-2020", fam: str = "T2-F3",
               assets=("UST_2Y", "UST_10Y"), horizons=(21, 63), tt: str = "level",
               asof: str = "2020-06-30", docs: bool = True, kind: str = "rates") -> pathlib.Path:
    u = root / uid
    (u / "panels").mkdir(parents=True, exist_ok=True)
    (u / "text").mkdir(exist_ok=True)
    (u / "card.toml").write_text(CARD.format(uid=uid, fam=fam, tt=tt, asof=asof,
                                             assets=list(assets), horizons=list(horizons)).replace("'", '"'),
                                 encoding="utf-8")
    make_panel(list(assets), asof, kind=kind).to_parquet(u / "panels" / f"{kind}_daily.parquet", index=False)
    if docs:
        (u / "text" / "fomc_statement_20200610.txt").write_text(textwrap.dedent("""
            The Committee decided to maintain the target range for the federal funds rate.
            Inflation pressures remain elevated and the Committee is prepared to tighten policy.
            Financial conditions show stress and volatility in Treasury markets.
            """), encoding="utf-8")
        (u / "text" / "future_doc_20210101.txt").write_text("A document from the future about rate cuts.",
                                                             encoding="utf-8")
        (u / "text" / "corpus_index.json").write_text(
            '{"documents": [{"doc_id": "fomc_statement_20200610", "timestamp": "2020-06-10", '
            '"doc_type": "fomc_statement", "file": "fomc_statement_20200610.txt"},'
            '{"doc_id": "future", "timestamp": "2021-01-01", "doc_type": "news", '
            '"file": "future_doc_20210101.txt"}]}', encoding="utf-8")
    return u


@pytest.fixture
def unit(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_unit(tmp_path)


@pytest.fixture
def single_unit(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_unit(tmp_path, uid="t2-F4-synthetic-2020", fam="T2-F4", assets=("UST_10Y",), horizons=(21,))


@pytest.fixture
def out_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


def no_llm() -> dict:
    return {"llm": {"enabled": False}}
