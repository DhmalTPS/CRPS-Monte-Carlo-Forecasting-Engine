import pytest

from agenthon.card import find_card, parse_card, validate_card
from agenthon.errors import CardError


def test_parse_card_fields(unit):
    c = parse_card(unit / "card.toml")
    assert c.unit_id == "t2-F3-synthetic-2020"
    assert c.family == "T2-F3"
    assert c.assets == ("UST_2Y", "UST_10Y")
    assert c.horizons == (21, 63)
    assert c.asof == "2020-06-30"
    assert c.n_cells == 4 and not c.is_single_cell
    assert c.effective_weights() == (0.5, 0.3, 0.2)


def test_single_cell_weights(single_unit):
    c = parse_card(single_unit / "card.toml")
    wm, wj, wt = c.effective_weights()
    assert wj == 0.0 and abs(wm - 0.5 / 0.7) < 1e-12 and abs(wt - 0.2 / 0.7) < 1e-12


def test_find_card_parent_layout(unit):
    assert find_card(unit / "panels") == unit / "card.toml"


def test_validate_uses_earlier_cutoff(unit):
    c = parse_card(unit / "card.toml")
    assert validate_card(c, "2020-06-01") == "2020-06-01"
    assert validate_card(c, "2021-01-01") == "2020-06-30"
    assert validate_card(c, "garbage") == "2020-06-30"


def test_missing_card_raises(tmp_path):
    (tmp_path / "a" / "panels").mkdir(parents=True)
    with pytest.raises(CardError):
        find_card(tmp_path / "a" / "panels", None)
