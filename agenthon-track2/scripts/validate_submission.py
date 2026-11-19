"""Run the OFFICIAL admissibility gates g0-g3 (installed qfbench2_track_forecasting) on outputs.

    python scripts/validate_submission.py --card <unit>/card.toml --out <dir-with-3-files>
    python scripts/validate_submission.py --units <units-dir> --outputs artifacts/forecasts

Requires: pip install qfbench2-common@v2.4.4 and the track repository (pip install -e <repo>).
Optional --realized-from offline/data scores against reconstructed outcomes (raw, unrankable).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib


def gates(card_path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    from qfbench2_track_forecasting.grid import grid_from_card
    from qfbench2_track_forecasting.normalization import NormalizationMode
    from qfbench2_track_forecasting.scoring import GATES, hydrate_ctx

    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    ctx = {"card": card, "output_dir": out_dir, "unit_dir": card_path.parent,
           "unit_handle": str(card["task"]["id"]), "expected_grid": grid_from_card(card),
           "grid_source": "card", "normalization_mode": NormalizationMode.RAW_UNRANKABLE,
           "ref_scale": None, "realized": None}
    hydrate_ctx(ctx)
    res = {}
    for name, g in GATES:
        r = g(ctx)
        res[name] = "pass" if r.passed else f"fail {r.label} {dict(r.detail)}"
        if not r.passed:
            break
    return {"unit": card["task"]["id"], "admissible": all(v == "pass" for v in res.values())
            and len(res) == len(GATES), "gates": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--units", type=pathlib.Path)
    ap.add_argument("--outputs", type=pathlib.Path)
    a = ap.parse_args()
    rows = []
    if a.card and a.out:
        rows.append(gates(a.card, a.out))
    elif a.units and a.outputs:
        for u in sorted(a.units.iterdir()):
            if (u / "card.toml").is_file():
                od = a.outputs / u.name
                rows.append(gates(u / "card.toml", od) if od.is_dir() else
                            {"unit": u.name, "admissible": False, "gates": {"missing": "no output"}})
    else:
        ap.error("give --card/--out or --units/--outputs")
    bad = [r for r in rows if not r["admissible"]]
    for r in bad:
        print(json.dumps(r))
    print(f"admissible {len(rows) - len(bad)}/{len(rows)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
