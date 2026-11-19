"""configs/default.yaml must mirror DEFAULTS exactly (a stale YAML silently overrides the code)."""

import pathlib

import yaml

from agenthon.config import DEFAULTS, load_config
from agenthon.resources import select_vintage, load_artifact, vintage_overrides

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_default_yaml_mirrors_defaults():
    y = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    assert y == DEFAULTS


def test_vintage_selection_is_strictly_before_asof():
    art = load_artifact("params_vintages.json")
    assert select_vintage(art, "2008-12-31") is None  # fit_cutoff must be < as-of
    assert select_vintage(art, "2009-01-02")["fit_cutoff"] == "2008-12-31"
    assert select_vintage(art, "2025-07-01")["fit_cutoff"] == "2024-12-31"
    assert vintage_overrides("2001-01-01") == ({}, "priors")


def test_family_yaml_loads():
    for f in ("T2-F1", "T2-F2", "T2-F3", "T2-F4"):
        assert f.lower().replace("t2-", "") in load_config(family=f)["families"]
