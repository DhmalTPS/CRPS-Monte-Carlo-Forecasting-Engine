"""Adversarial LLM stress over real units: inject a fake House client with EXTREME / contradictory /
garbage answers and check that every unit still (a) runs the full path, (b) passes the official gates,
(c) keeps its spread within sane bounds of M0 (text cannot blow a card up).

    python scripts/llm_stress.py --units <units-dir> --mode extreme|contradict|garbage
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class FakeHouse:
    def __init__(self, assets: list[str], mode: str):
        self.calls, self.available, self.mode, self.assets = 0, True, mode, assets

    def chat(self, messages, max_tokens=None, deadline=None):
        self.calls += 1
        if self.mode == "garbage":
            return "Sure! Rates will go up a lot, maybe 3%. {not json"
        up = 0.97 if self.mode == "extreme" else 0.03
        a = {x: {"p_up": up, "magnitude": "extreme", "vol_view": "much_higher" if self.mode == "extreme"
                 else "lower", "tail_side": "both"} for x in self.assets}
        sc = [{"name": "shock", "probability": 0.5, "direction": {x: 1 for x in self.assets},
               "magnitude": "extreme", "vol_view": "much_higher"},
              {"name": "reverse", "probability": 0.3, "direction": {x: -1 for x in self.assets},
               "magnitude": "extreme", "vol_view": "much_higher"},
              {"name": "calm", "probability": 0.2, "direction": {x: 0 for x in self.assets},
               "magnitude": "weak", "vol_view": "lower"}]
        return json.dumps({"assets": a, "scenarios": sc, "persistence": "persistent",
                           "regime_evidence": "supports_stress", "confidence": 1.0, "novelty": 1.0,
                           "established": True, "cited_docs": [], "summary": "stress test"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=pathlib.Path, required=True)
    ap.add_argument("--mode", default="extreme", choices=["extreme", "contradict", "garbage"])
    a = ap.parse_args()
    from agenthon.card import parse_card
    from agenthon.main import run
    from validate_submission import gates

    rows = []
    for u in sorted(a.units.iterdir()):
        if not (u / "card.toml").is_file():
            continue
        card = parse_card(u / "card.toml")
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td)
            path, res = run(u / "panels", u / "text", None, out / "forecast.parquet",
                            client=FakeHouse(list(card.assets), a.mode))
            ok = gates(u / "card.toml", out)["admissible"]
            d = res.diagnostics if res else {}
            ratio = float("nan")
            if res is not None:
                ratio = float(np.max(res.draws.std(0) / np.maximum(d["m0_draws"].std(0), 1e-12)))
            rows.append({"unit": u.name, "path": path, "admissible": ok, "gate": (d.get("text") or {}).get("gate"),
                         "source": (d.get("text") or {}).get("source"), "max_sd_ratio_vs_m0": ratio,
                         "mechanism": (d.get("text") or {}).get("mechanism")})
    df = pd.DataFrame(rows)
    print(df.groupby(["path", "source"]).size().to_string())
    print(f"mode={a.mode}: admissible {int(df.admissible.sum())}/{len(df)}; full path {int((df.path == 'full').sum())}; "
          f"gate mean {df.gate.mean():.3f} max {df.gate.max():.3f}; max sd ratio vs M0: median "
          f"{df.max_sd_ratio_vs_m0.median():.2f}, max {df.max_sd_ratio_vs_m0.max():.2f}")
    print(df.sort_values("max_sd_ratio_vs_m0").tail(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
