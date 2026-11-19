# Offline data and the full reproduction chain

## Data
`build_history.py` rebuilds continuous panels from the union of the published practice units'
truncated copies (tuning on published practice data is permitted). Outputs:

| file | content | coverage |
|---|---|---|
| `rates_daily_history.parquet` | UST 2/5/7/10/20/30Y, % p.a. | 2000-01-03 … 2024-12-18 |
| `g10_fx_daily_history.parquet` | 10 G10 spot mids | 2000-01-03 … 2024-10-31 |
| `factors_daily_history.parquet` | MKT SMB HML MOM BAB QMJ, daily simple returns | 2000-01-03 … 2024-05-31 |
| `em_transfer_early_history.parquet` | BRL CNY INR early windows + as-of rows | 1993 … 2018 |
| `macro_monthly_vintages.parquet` | macro panel per vintage (as published on each card's as-of) | 4 vintages |

Audit (`manifests/history_manifest.json`): 0 conflicting (asset, date) values across units once
the worked exemplar (a non-FRED illustrative 2024 panel) is excluded.

**Deliberately not downloaded:** any 2025+ market data. The sealed window is H2 2025 – Q2 2026;
tuning on data after an evaluation task's cutoff would breach the artifact policy.

## Reproduction chain
```
python offline/data/build_history.py --units <track2-forecasting-public/units> --out offline/data
python offline/walk_forward/generate_cutoffs.py                  # 3,896 cards
python -m offline.ablations.M0_baseline 8                        # must print 1.0000
python -m offline.ablations.M1_random_walk 8
python -m offline.ablations.M2_statistical 8
python offline/experiments/tune.py --jobs 8 --prefix R2_ --base '{"numerical":{"lr_omega":0.0}}'
python offline/rl/train_policy.py --run R2_base                  # OOS policy table
python offline/walk_forward/run.py --name M11_full --overrides '{"_vintaged": true}'
python offline/experiments/fit_calibration.py --run M11_full     # M10 (rejected)
python offline/walk_forward/practice_eval.py --rung M11_full     # text rungs (lexicon)
python offline/walk_forward/practice_actions.py                  # family actions (≤2024 vintage)
python offline/walk_forward/aggregate.py M0_baseline M1_random_walk M11_full --save ablation_ladder
python offline/walk_forward/stress.py M11_full
```
Tables land in `offline/reports/tables/`; run parquet files in `offline/experiments/runs/`.
