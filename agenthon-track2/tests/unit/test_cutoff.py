"""L1 defence-in-depth: nothing dated after the as-of reaches the pipeline."""

import pandas as pd

from agenthon.card import parse_card
from agenthon.config import load_config
from agenthon.data import point_in_time_filter
from agenthon.data.availability import date_from_filename, doc_available


def test_future_document_is_dropped(unit):
    c = parse_card(unit / "card.toml")
    pit = point_in_time_filter(c, unit / "panels", unit / "text", c.asof, load_config())
    assert all(d.timestamp <= c.asof for d in pit.docs)
    assert pit.dropped_docs == 1
    assert [d.doc_id for d in pit.docs] == ["fomc_statement_20200610"]


def test_panel_rows_truncated_at_earlier_cutoff(unit):
    c = parse_card(unit / "card.toml")
    pit = point_in_time_filter(c, unit / "panels", unit / "text", "2020-05-29", load_config())
    for h in pit.histories.values():
        assert h.series.index.max() <= pd.Timestamp("2020-05-29")
    assert pit.dropped_rows > 0


def test_undated_documents_excluded():
    assert not doc_available(None, "2020-01-01")
    assert not doc_available("2020-01-02", "2020-01-01")
    assert doc_available("2020-01-01", "2020-01-01")
    assert date_from_filename("fomc_statement_20220316.txt") == "2022-03-16"
    assert date_from_filename("bis_hu_2010-07-15.txt") == "2010-07-15"
