"""Reconstruct continuous 2000-2024 panels for OFFLINE research from the published practice units.

The full panels (data/rates, data/fx, ...) are not shipped; each unit carries its own panel cut at
its own as-of. The union of those truncated copies is the full published history, and tuning on
published practice data is explicitly permitted by the rules. Nothing produced here is ever baked
into the submission image as a lookup: it is used for walk-forward model selection only, and every
fitted parameter is recorded with the cutoff of the data it was fitted on (vintaged artifacts).

Consistency is audited: when two units disagree on an (asset, date) value the conflict is counted
and the value from the unit with the LATEST as-of is kept (market closes are not revised; macro
panels are vintage-dependent, so macro is kept per-vintage separately).

We deliberately do NOT download post-2024 data: the sealed evaluation window is H2 2025 - Q2 2026,
and tuning on data after an evaluation task's cutoff would breach the artifact policy.

Usage:  python offline/data/build_history.py --units <track2-forecasting-public/units> --out offline/data
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tomllib

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, type=pathlib.Path)
    ap.add_argument("--out", default=pathlib.Path(__file__).parent, type=pathlib.Path)
    a = ap.parse_args()
    frames: dict[str, list[pd.DataFrame]] = {}
    macro_vintages: list[pd.DataFrame] = []
    for u in sorted(a.units.iterdir()):
        card = u / "card.toml"
        if not card.is_file() or u.name.startswith("t2-EXAMPLE"):
            continue  # the worked exemplar ships a non-FRED illustrative 2024 panel
        asof = tomllib.loads(card.read_text(encoding="utf-8"))["provenance"]["data_cutoff"]
        for p in sorted(u.rglob("*.parquet")):
            df = pd.read_parquet(p)
            col = "asset" if "asset" in df.columns else "asset_id"
            df = df.rename(columns={col: "asset"})[["date", "asset", "value"]].copy()
            df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
            df["unit_asof"] = asof
            df["unit"] = u.name
            if p.stem == "macro_monthly":
                macro_vintages.append(df)
            elif p.stem == "em_transfer_early":
                frames.setdefault("em_transfer_early", []).append(df)
            else:
                frames.setdefault(p.stem, []).append(df)
    report: dict[str, dict] = {}
    a.out.mkdir(parents=True, exist_ok=True)
    for stem, parts in frames.items():
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby(["asset", "date"])["value"]
        conflicts = int((g.nunique() > 1).sum())
        best = (df.sort_values("unit_asof").groupby(["asset", "date"], as_index=False).last())
        best = best[["date", "asset", "value"]].sort_values(["asset", "date"])
        best.to_parquet(a.out / f"{stem}_history.parquet", index=False)
        report[stem] = {
            "rows": int(len(best)),
            "assets": sorted(best["asset"].unique().tolist()),
            "start": str(best["date"].min().date()),
            "end": str(best["date"].max().date()),
            "conflicting_asset_dates": conflicts,
        }
    if macro_vintages:
        mv = pd.concat(macro_vintages, ignore_index=True)
        mv.to_parquet(a.out / "macro_monthly_vintages.parquet", index=False)
        report["macro_monthly"] = {"vintages": sorted(mv["unit_asof"].unique().tolist()),
                                   "rows": int(len(mv))}
    (a.out / "manifests").mkdir(exist_ok=True)
    (a.out / "manifests" / "history_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
