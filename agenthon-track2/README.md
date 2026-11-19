# agenthon-track2 — a fail-safe probabilistic forecasting engine with a gated LLM

Agenthon 2026 Track 2 (reasoning-augmented time-series forecasting). One Docker verb,
`forecast`, turns a unit (card + point-in-time panels + dated corpus) into
`forecast.parquet` (joint Monte-Carlo draws), `forecast_meta.json` and
`forecast_rationale.md`.

The design principle, validated rather than assumed:

> **The score is a per-card RATIO to a text-blind baseline (M0).** Beating it is a decision
> problem under a ratio loss, not a "better point forecast" problem. The numerical prior is
> strong but humble; text is a gated, bounded adjustment; and the submitted draws are the
> metric-optimal act derived from the honest belief.

## Quick start

```bash
pip install -e ".[dev]"                # numpy, pandas, pyarrow, pyyaml (+ pytest)
forecast --panels <unit>/panels/ --text <unit>/text/ --asof YYYY-MM-DD --out out/forecast.parquet
python scripts/validate_submission.py --card <unit>/card.toml --out out   # official gates g0-g3 (needs the organiser scorer, see below)
python -m pytest                       # 84 tests: unit, integration, adversarial
```

The gate validator (and the last step of `submission/run.sh`) imports the organiser scorer:
`pip install qfbench2-common@v2.4.4` and `pip install -e <track2-forecasting-public>`.

Image: `docker build --platform linux/amd64 -t agenthon-t2:1.0.1 .` then
`bash submission/run.sh <unit-dir> agenthon-t2:1.0.1` (offline, read-only, uid 65534, 64 MiB /tmp, 256 PIDs —
the Development runtime's exact restrictions). Upload: `python scripts/run_submission.py
--image <registry>/<repo>@sha256:<digest>` (validates and seals the descriptor with the
toolkit), then `qfbench2 submission pack ...`.

## What is verified (this repository, 2026-09-29, v1.0.1)

| Check | Result |
|---|---|
| Official gates g0-g3 (installed `qfbench2_track_forecasting`) on all 104 practice units | **104/104 admissible**, all on the full path |
| Same, under the image's pinned stack (Py 3.13.5, numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0) | 104/104, 84/84 tests |
| M0 replica vs the organiser's `docs/M0-BASELINE.md` worked example (track2-forecasting-public) | seed 795546941, 128 steps, horizon rule 24→21, monthly 8/9, NFP 2 |
| Submitting the M0 replica in the walk-forward | **exactly 1.0000** on 3,896 cards (normalisation chain verified) |
| Metric parity vs `qfbench2_common` (CRPS / variogram / pinball) | |Δ| ≤ 3e-16 |
| House HTTP contract (mock server) | `/v1/chat/completions`, Bearer, n=1, thinking off, no tools; 503 → lexicon |
| Adversarial LLM stress on all 104 units (extreme / contradictory / garbage answers) | 104/104 admissible; text widening capped at 2x; a text update failing spread sanity degrades to numeric-only |
| Container, harness-exact (read-only, uid 65534, no network, 64 MiB /tmp) | 104/104 admissible (`scripts/docker_smoke.ps1 -All`) |
| Runtime, numeric path | 0.38 s mean, 0.64 s max per unit (+ ≤ 2 × 120 s for House calls) |

## Results (walk-forward, production-exact, parameters out of sample by vintage)

3,896 synthetic cards on the reconstructed 2000-2024 panels, shaped like the practice roster.
Every card uses only parameters fitted on outcomes known BEFORE its as-of (vintaged
artifacts), so every fitted parameter is out of sample; the policy pessimism and the prior λ
were chosen on this walk-forward. Lower is better; 1.0 = M0.

| rung | mean | 95% CI (quarter-block bootstrap) | note |
|---|---|---|---|
| M0 baseline (replica) | 1.000 | — | sanity |
| M1 driftless random walk | 1.048 | [1.018, 1.078] | dropping M0's drift alone *loses* under a ratio |
| M2 vol term structure (EWMA→LR, VR) | 1.064 | [1.036, 1.094] | a better honest belief does not beat M0 |
| **M11 full system (ships)** | **0.968** | **[0.952, 0.985]** | score-aware (γ, λ) + safe policy |
| M11 + PIT calibration (M10) | 0.971 | [0.956, 0.991] | rejected: worse OOS (0.964 → 0.977 after 2015) |

By vintage: 2009-2014 cards (≤2008 vintage) **0.936**; 2015-2024 cards
(≤2014 vintage + policy) **0.964**; 2003-2008 cards use theory priors only (1.006).

Stress windows (M11): taper 0.954, COVID 0.985, GFC 1.013, 2022 hikes 1.062 — the system is
weakest exactly where cards are event-selected, which is why the ≤2024 vintage widens F2/F4
(below).

Practice units (event-selected; realised values reconstructed from the published history,
evaluation only): numeric system by family F1 0.906 → F3 0.942 at the generic action;
F2/F4 do not beat M0 at the generic action (best 0.993 for F2; F4 1.010, or 0.97 with the
belief widened ×1.4). Lexicon-only
text is neutral (Δtext +0.001) — as designed, a capped fallback. The LLM's Δtext cannot be
measured offline (no House endpoint off-platform); the gate bounds its downside.

## The five findings that shaped the architecture

1. **Convexity of the ratio.** Per card, gains are bounded (ratio ≥ ~0.3) and losses reach
   4.0. Moving the centre away from M0's (even removing its noisy drift) costs more on the
   cards where M0 was right than it earns elsewhere (M1: 1.048).
2. **The metric-optimal submission is a reweighted belief.** For a proper score S and outcome
   weight w(y) = 1/S_M0(y), argmin_Q E_P[w S(Q,y)] = P reweighted by w. We submit the honest
   belief resampled with w^γ (tempered), with a hard floor: no cell narrower than 60 % of the
   honest sd. Simulation (M0 exactly right): 1.004 → 0.962.
3. **Model averaging with the baseline caps blow-ups.** CRPS is convex in the forecast, so
   mixing a fraction λ of M0-law draws gives ratio ≤ (1-λ)·ratio + λ. Joint (γ, λ): 1.049 →
   0.955 on the same belief.
4. **Pessimism is what makes RL work in noisy finance.** A contextual policy over (γ, λ)
   overfits at low pessimism (test 0.969 vs 0.9665 default) and helps only with an LCB rule
   (pessimism 1-2: 0.964-0.965; `offline/reports/tables/policy_eval.csv`). Offline rewards are full-information, so the policy is cost-sensitive
   contextual learning with safe policy improvement — not PPO.
5. **Event selection.** Practice cards are authored around shocks; uniform-date validation
   understates tail risk. F4/F2 overrides (belief ×1.35 / ×1.2, heavier M0 hedge) enter only
   the ≤2024 vintage, because they were chosen on practice outcomes up to 2024.

## Pipeline (L0 → L23)

```
card.toml ─ L0 card/          parse, validate, trusted as-of, family policy
          ─ L1 data/          point-in-time filter (defence in depth), per-unit corpus only
          ─ L2-L4 numerical/  steps (gap rule), M0 horizon rule, state, soft 3-regime
          ─ M0  random_walk   exact baseline replica  ──► fail-safe (scores 1.0) + reference
          ─ L5-L7 ensemble    EWMA→long-run variance term structure × shrunk variance ratio
          ─ L20 tails/        in-unit backtest + EB shrinkage (available; off: no OOS gain)
          ─ policy/           context → (γ, λ) by per-action Bayesian ridge + LCB
          ─ L8-L12 text/      passages → dedup (5-gram Jaccard × time) → state-conditioned
                              query → BM25 × recency/asset/novelty (relevance-gated) → MMR
          ─ L13 llm           ONE schema-constrained House call (thinking off, T=0, seed),
                              retry once on parse failure; else lexicon (capped) / neutral
          ─ L14-L19 fusion/   κ contradiction, monotone logistic gate (g=0 ⇒ P1≡P0),
                              tilt / mixture / one-sided tail, PSD-safe dependence blend
          ─ L21 sampling/     path increments × correlation, t-mixing, soft ZLB floor
          ─ L22 calibration/  PIT maps (implemented; disabled by ablation)
          ─ L22b decision/    score-aware resampling (γ) + M0 model averaging (λ)
          ─ L23 output/       atomic write of exactly 3 files, local mirror of g0-g3
any exception ─ safety/       exact M0 draws → minimal walk → constant (never a missed card)
```

## Rules compliance (highlights)

* House model only via `MODEL_ENDPOINT/v1/chat/completions` + `MODEL_TOKEN`; ≤ 3 calls/unit
  of the 25 allowed; ≤ 4000 output tokens; no vendor tools; temperature 0 and seed pinned.
* No pretrained neural checkpoints (artifact policy): text is lexical (BM25/dictionaries);
  `numerical/foundation.py` is a documented no-op. No GPU is needed: the numeric
  path is 0.4 s/card on CPU and the image must run without a device.
* No cross-unit lookup: retrieval and analogues use the mounted unit only; nothing derived
  from a unit's target is baked into the image. Practice outcomes are used for evaluation and
  for two family-level scalars in the ≤2024 vintage (tuning, which the rules permit).
* Every fitted artifact carries a `fit_cutoff`; a card uses a vintage only if
  fit_cutoff < as-of (see `ARTIFACT_PROVENANCE.md`).
* `/output` receives exactly three files; logs go to stderr.

## Layout

`src/agenthon/` production (what the image runs) · `configs/` priors, profiles, families ·
`prompts/` system + task prompts + LLM output schema · `offline/` walk-forward, ablations M0-M12,
scoring, RL training, experiments, reports · `tests/` unit / integration / adversarial ·
`scripts/` local runs, gate validation, submission prep · `submission/` image wrappers.

Reproduce everything: see `offline/data/README.md` (history rebuild → cards → rungs →
tuning → policy → vintages).

## Audit log

* **1.0.1 (2026-09-29)** — full audit. Fixed: (1) a text update that failed the spread sanity
  check discarded the numeric forecast too (fell to M0); it now degrades to numeric-only;
  (2) text-driven widening capped at 2x per cell and max centre shift lowered (alpha 0.35 → 0.25,
  the ratio loss is convex); (3) House HTTP 400/422 now retries once with the bare OpenAI body
  (for a deployment rejecting `seed`/`chat_template_kwargs`); (4) last-resort constant fallback
  anchors on the level stated in the card prose instead of 0; (5) rationale states lambda;
  (6) removed four unused config keys; provenance feature count corrected (16). Earlier fixes
  from testing: unsafe LLM coercions (`"established": "yes"`, confidence 250), irrelevant recent
  passages outranking relevant ones, stale `default.yaml` overriding code defaults, pandas-3
  datetime resolution in day counts.
