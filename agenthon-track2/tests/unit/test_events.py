from agenthon.text.dedup import deduplicate
from agenthon.text.events import extract_events
from agenthon.types import TextDoc


def _doc(i, ts, body):
    return TextDoc(f"d{i}", ts, "fomc_statement", "", "Federal Reserve", body)


BODY = ("The Committee decided to raise the target range for the federal funds rate. Inflation "
        "pressures remain elevated and further tightening may be appropriate given upside risks to inflation.")


def test_events_carry_tone_and_assets():
    ev = extract_events([_doc(1, "2022-03-16", BODY)], ["UST_2Y"], [])
    assert ev and ev[0].hawk > 0 and "UST" in ev[0].assets


def test_wire_copies_collapse_to_one_event():
    docs = [_doc(i, "2022-03-16", BODY) for i in range(5)] + [_doc(9, "2022-09-01", BODY)]
    ev = deduplicate(extract_events(docs, ["UST_2Y"], []), 0.55, 5)
    # five same-day copies -> one cluster; the September recurrence is a separate event
    assert len(ev) == 2
    assert max(e.corroboration for e in ev) == 5
    assert min(e.novelty for e in ev) < 1.0
