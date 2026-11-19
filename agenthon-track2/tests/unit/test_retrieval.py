from agenthon.text.rerank import rerank_and_compress
from agenthon.text.retrieval import retrieve
from agenthon.types import Event


def _ev(i, ts, text, assets=("UST",)):
    return Event(f"e{i}", f"d{i}", ts, "fomc_statement", "Federal Reserve", text, (), assets, 0.0, 0.0, 0)


def test_relevance_and_recency_rank_first():
    evs = [_ev(1, "2010-01-01", "treasury yields federal funds tightening", ()),
           _ev(2, "2022-06-10", "treasury yields federal funds tightening"),
           _ev(3, "2022-06-10", "weather report sunny")]
    r = retrieve(evs, ["treasury", "yields", "federal funds"], "2022-06-15", ["UST_2Y"],
                 {"bm25": 1, "recency": 0.6, "asset": 0.8, "novelty": 0.3, "doc_type": 0.4})
    assert r[0].event_id == "e2" and r[-1].event_id == "e3"


def test_rerank_respects_budget_and_diversity():
    evs = [_ev(i, "2022-06-10", "treasury yields federal funds tightening " * 30) for i in range(10)]
    for i, e in enumerate(evs):
        e.score = 1.0 - i * 0.01
    out = rerank_and_compress(evs, top_k=5, budget_chars=2000)
    assert sum(len(e.text) for e in out) <= 2000 and 1 <= len(out) <= 5
