"""Empty or missing corpus is a valid state: g = 0 and the forecast is the numerical prior."""

import shutil

from agenthon.main import run


def test_missing_text_dir(unit, out_dir):
    shutil.rmtree(unit / "text")
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                    overrides={"llm": {"enabled": False}})
    assert path == "full" and res.diagnostics["text"]["gate"] == 0.0


def test_empty_documents(unit, out_dir):
    for p in (unit / "text").iterdir():
        if p.suffix == ".txt":
            p.write_text("   \n", encoding="utf-8")
    path, res = run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                    overrides={"llm": {"enabled": False}})
    assert path == "full" and res.diagnostics["text"]["gate"] == 0.0
