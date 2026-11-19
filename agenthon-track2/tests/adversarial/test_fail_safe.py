"""Any exception inside the pipeline -> the exact M0 replica is written (scores 1.0), exit clean."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

import agenthon.main as main_mod
from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.numerical import build_grid, m0_forecast


def test_crash_writes_exact_m0(monkeypatch, unit, out_dir):
    def boom(*a, **k):
        raise RuntimeError("synthetic crash deep in the pipeline")

    monkeypatch.setattr(main_mod, "forecast_card", boom)
    path, res = main_mod.run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet")
    assert path == "failsafe_m0" and res is None
    df = pd.read_parquet(out_dir / "forecast.parquet")
    c = parse_card(unit / "card.toml")
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, load_config())
    g = build_grid(c, pit.histories)
    m0 = m0_forecast(c, g, pit.histories).draws
    wide = df.pivot_table(index="draw", columns=["asset", "horizon"], values="value")
    ours = np.column_stack([wide[(cell.asset, cell.horizon)].to_numpy() for cell in g.cells])
    assert np.allclose(ours, m0)  # bit-for-bit the baseline's draws
    meta = json.loads((out_dir / "forecast_meta.json").read_text())
    assert meta["unit_id"] == c.unit_id and meta["n_draws"] == 500


def test_keyboard_interrupt_still_writes(monkeypatch, unit, out_dir):
    def ki(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_mod, "forecast_card", ki)
    path, _ = main_mod.run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet")
    assert path == "failsafe_m0" and len(list(out_dir.iterdir())) == 3


def test_text_blowup_degrades_to_numeric_only(monkeypatch, unit, out_dir):
    import agenthon.fusion.distribution as dist

    real = dist.distribution_update

    def huge(p0, *a, **k):
        p1, led = real(p0, *a, **k)
        p1 = p1.copy()
        for s in p1.scenarios:
            s.scale = s.scale * 50.0  # simulate a text update that blows the spread up
        return p1, led

    monkeypatch.setattr(main_mod, "distribution_update", huge)
    path, res = main_mod.run(unit / "panels", unit / "text", None, out_dir / "forecast.parquet",
                             overrides={"gate": {"w0": 5.0}, "llm": {"enabled": False}})
    assert path == "numeric_only" and any(r.get("step") == "text_rejected" for r in res.ledger)


def test_constant_fallback_uses_prose_anchor(tmp_path):
    from agenthon.card import parse_card
    from agenthon.safety.fail_safe import fail_safe_forecast
    from tests_helpers import write_transfer_card

    u = write_transfer_card(tmp_path)
    card = parse_card(u / "card.toml")
    out = tmp_path / "o"
    out.mkdir()
    path = fail_safe_forecast(card, u / "nopanels", out / "forecast.parquet", RuntimeError("x"))
    df = pd.read_parquet(out / "forecast.parquet")
    assert path == "failsafe_constant" and abs(df.value.median() - 6.2097) < 0.5
