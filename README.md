# CRPS Monte Carlo Forecasting Engine

A point-in-time probabilistic forecaster that emits joint Monte Carlo draws for rates, FX, equity factors and macro series, fuses gated text signals into a numerical prior, and chooses the submitted distribution as the optimal act under a ratio-to-baseline proper-score loss.

## Context

- **Event:** Agenthon 2026 (a NeurIPS 2026 competition), Track 2: reasoning-augmented probabilistic time-series forecasting, built on the QFBench 2.0 benchmark and scored on CodaBench.
- **Dates:** built September 2026; Development upload 2026-09-29; Final + Verification phase 13-25 October 2026.
- **Team:** solo (registered as "Team 89").
- **Authorship:** the research design, evaluation protocol and final selection are mine. I used AI coding assistants for parts of the implementation and for adversarial review.
- **Organiser material (not included here):** [Agenthon-2026/Agenthon2026-public](https://github.com/Agenthon-2026/Agenthon2026-public) (rules, starter packs) and [Agenthon-2026/track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public) (practice units, official scorer `qfbench2_track_forecasting`).

## Problem

Each task ("card") gives an as-of date, a set of assets and horizons (mostly 21/63/126/189 business days), point-in-time numeric panels (UST 2-30Y yields, G10 FX, equity factors MKT/SMB/HML/MOM/BAB/QMJ, EM FX, monthly CPI/NFP) and a dated text corpus (central-bank statements and similar). The submission must return 200-20,000 **joint** Monte Carlo draws. It runs in an offline, read-only Docker container; the only network access is an organiser-hosted LLM.

Scoring is a composite of marginal CRPS (50%), joint variogram score (30%) and a tail/pinball penalty (20%) (organiser-defined weights), normalised **per card as a ratio to the organiser's text-blind random-walk baseline M0**: 1.0 = baseline, lower is better, clipped at 4.

## Approach

1. **Strict as-of discipline.** Panel rows after the as-of and undated or future documents are dropped (`src/agenthon/data/availability.py`, tested in `tests/unit/test_cutoff.py`). Every fitted parameter is *vintaged*: a card may only use a vintage whose `fit_cutoff < as-of` (vintages 2008-12-31, 2014-12-31, 2024-12-31; see `ARTIFACT_PROVENANCE.md`).
2. **Numerical prior.** EWMA-to-long-run variance term structure with shrunk variance ratios, plus an exact replica of the M0 baseline that doubles as the fail-safe.
3. **Joint Monte Carlo sampler.** Correlated Gaussian path increments (Cholesky), a shared t-mixing scale, antithetic pairs and a soft zero-lower-bound floor (`src/agenthon/sampling/`).
4. **Text path, bounded by construction.** BM25 retrieval with recency, de-duplication and MMR; one schema-constrained LLM call (at most 3 with retries; temperature 0, seed pinned) with a lexicon fallback; a monotone logistic information gate where g = 0 makes the text-on forecast identical to the text-blind one; text-driven widening capped at 2x.
5. **Score-aware decision layer.** Under a ratio loss, a better honest forecast is not the optimal submission. For a proper score S and outcome weight w(y) = 1/S_M0(y), the optimal report is the belief reweighted by w. The engine resamples the belief with weights w^γ (tempered, with a 60% spread floor) and mixes in a fraction λ of M0 draws, which caps blow-ups because CRPS is convex in the forecast. (γ, λ) is chosen per card by a contextual Bayesian-ridge policy with a lower-confidence-bound rule. Derivations are in the report appendix.
6. **Never miss a card.** Any exception falls back to exact M0 draws, then a minimal walk, then a constant.

## Results

All scores are ratios to M0 (1.0 = baseline, lower is better).

| Result | Value | Source |
|---|---|---|
| **Official Development leaderboard score** (v1.0.1, 71/71 cards scored) | **0.9782** | `docs/report/sections/12_results.tex`; per-card table `offline/reports/leaderboard/dev_upload1_v1.0.1.csv` |
| By family (official Dev) | F1 0.895, F2 0.965, F3 0.947, **F4 1.078** | same |
| **Walk-forward, full system M11** (3,896 synthetic cards; parameters fitted out of sample by vintage, design choices made on the same walk-forward) | **0.968, 95% CI [0.952, 0.985]** (quarter-block bootstrap) | `offline/reports/tables/ablation_ladder.csv` |
| Final-phase leaderboard rank | 38 / 87 | final results (not recorded in this repository) |

Walk-forward ablation ladder (same 3,896 cards):

| Rung | Mean | 95% CI |
|---|---|---|
| M0 baseline replica | 1.000 | exact |
| M1 driftless random walk | 1.048 | [1.018, 1.078] |
| M2 variance term structure | 1.064 | [1.036, 1.094] |
| **M11 full system (shipped)** | **0.968** | **[0.952, 0.985]** |
| M11 + PIT calibration (M10) | 0.971 | [0.956, 0.991] |

By vintage (fitted parameters out of sample; the policy pessimism and the prior λ were chosen on this walk-forward): 2009-2014 cards 0.936; 2015-2024 cards 0.964; 2003-2008 cards (theory priors only) 1.006.

**What did not work, or is not established:**

- **Better honest forecasts lose under the ratio loss.** M1 and M2 improve the belief but score 1.048 and 1.064; almost all of the gain comes from the decision layer, not from a sharper point forecast.
- **Stress windows:** taper 0.954, COVID 0.985, but **GFC 1.013 and 2022 hikes 1.062**, i.e. worse than the baseline exactly where cards are event-selected. The tail/shock family F4 is also worse than baseline on the official board (1.078).
- **The text/LLM contribution is not statistically distinguishable from zero.** Official (LLM on) minus local (lexicon fallback, no LLM) on the same 65 cards: -0.0095, 95% CI [-0.026, +0.009]. The lexicon-only fallback is neutral (Δ +0.001).
- **PIT calibration was rejected**: worse out of sample after 2015 (0.964 to 0.977). Fat-tailed beliefs (ν = 7) were the worst belief variant.
- **RL policy overfits without pessimism**; it helps only with an LCB rule.
- On the 65 Dev cards the local evaluator can reconstruct, the official score is 0.983 and the local reconstruction 0.992 (per-card correlation 0.912). The F2/F4 family overrides in the 2024 vintage were tuned on practice outcomes and are in-sample.
- 75 of the 103 practice cards leak their realised values through sibling units' panels, so the practice board was not used as a tuning signal.

## Repository layout

```
agenthon-track2/
  src/agenthon/      production pipeline run by the image (card, data, numerical, text, fusion,
                     sampling, decision, policy, tails, calibration, safety, output); resources/ holds
                     the vintaged parameter, policy and calibration tables
  configs/           priors, Development/Final profiles, family overrides
  prompts/           system and task prompts, LLM output JSON schema
  offline/           walk-forward generator and runner, ablation ladder M0-M12, scoring mirrors,
                     policy training, tuning, history rebuild script; reports/tables/*.csv results
  tests/             84 tests: unit, integration, adversarial (bad JSON, NaNs, timeouts, LLM failure)
  scripts/           local runs, official-gate validation, Docker smoke, LLM stress, submission prep
  submission/        harness-exact Docker wrappers and descriptor template
  docs/report/       technical report (LaTeX sources and compiled PDF, with proofs)
  Dockerfile, ARTIFACT_PROVENANCE.md, AGENTS.md
```

Not included: the organiser practice units and the history panels rebuilt from them (`offline/data/*.parquet`; the competition rules say not to redistribute task data), walk-forward run files (`offline/experiments/runs/*.parquet`, ~40 MB, regenerable), generated forecasts, the submission descriptors, and the organiser repositories.

## How to run

Requires Python 3.11+ (developed on 3.13).

```bash
cd agenthon-track2
pip install -e ".[dev]"            # numpy, pandas, pyarrow, pyyaml (+ pytest)
python -m pytest                   # 84 tests; 9 skip without the organiser scorer and a sibling _reference/track2-forecasting-public checkout

# forecast one practice unit from track2-forecasting-public/units/<unit>
forecast --panels <unit>/panels/ --text <unit>/text/ --asof YYYY-MM-DD --out out/forecast.parquet
# the official gates need the organiser scorer: pip install qfbench2-common@v2.4.4
# and pip install -e <track2-forecasting-public> (provides qfbench2_track_forecasting)
python scripts/validate_submission.py --card <unit>/card.toml --out out   # official gates g0-g3

# container, with the harness's restrictions (offline, read-only, uid 65534)
docker build --platform linux/amd64 -t agenthon-t2:1.0.1 .
bash submission/run.sh <unit-dir> agenthon-t2:1.0.1
```

The LLM path reads `MODEL_ENDPOINT` and `MODEL_TOKEN` from the environment; without them it falls back to the lexicon signal. The full offline reproduction chain (history rebuild, 3,896 walk-forward cards, ablations, tuning, policy, stress) is in `agenthon-track2/offline/data/README.md`; it needs a local clone of `track2-forecasting-public`.

## Notes and limitations

- 0.9782 is the **Development-phase** score on 71 historical practice cards (2003-2024). The sealed Final window is H2 2025 to Q2 2026; the Final score is not recorded here.
- The walk-forward cards are synthetic (uniform dates, every 21 business days from 2003), so they under-sample the shock-centred cards the organisers write; this is the main reason the stress and F4 numbers are weaker.
- The value of the LLM is essentially unmeasured offline: the organiser endpoint exists only inside the scoring environment.
- The compiled report PDF (`docs/report/agenthon_track2_report.pdf`) predates a correction made in the LaTeX source (`12_results.tex`) and this README: under the Development configuration the 65 reconstructable Dev cards score 0.992 locally (0.983 is their official score). Its policy section also quotes a ridge setting (λ_r = 100) whose run is not committed; the committed `offline/reports/tables/policy_eval.csv`, which `agenthon-track2/README.md` quotes, gives test 0.969 at pessimism 0 and 0.964-0.965 at pessimism 1-2 (vs 0.9665 default).
- See `docs/report/sections/14_limitations.tex` for the full list.
