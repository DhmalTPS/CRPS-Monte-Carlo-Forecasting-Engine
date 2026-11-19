"""Full pipeline on real practice units + the OFFICIAL gates (skipped without the reference repo)."""

import pathlib

import pytest

from agenthon.main import run

REF = pathlib.Path(__file__).resolve().parents[3] / "_reference" / "track2-forecasting-public" / "units"
UNITS = ["t2-EXAMPLE-ust-curve-1m", "t2-F1-cpi-glidepath-2023", "t2-F2-cnh-stress-2015",
         "t2-F3-dollar-squeeze-2020", "t2-F4-quant-crowding-2007", "t2-F3-cpi-shock-cross-2022"]


@pytest.mark.skipif(not REF.is_dir(), reason="reference units not checked out")
@pytest.mark.parametrize("name", UNITS)
def test_practice_unit_passes_official_gates(name, tmp_path):
    gates = pytest.importorskip("qfbench2_track_forecasting.scoring")
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from validate_submission import gates as run_gates

    u = REF / name
    out = tmp_path / "out"
    out.mkdir()
    path, _ = run(u / "panels", u / "text", None, out / "forecast.parquet", overrides={"llm": {"enabled": False}})
    assert path == "full"
    r = run_gates(u / "card.toml", out)
    assert r["admissible"], r
