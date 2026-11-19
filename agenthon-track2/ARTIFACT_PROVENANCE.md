# ARTIFACT_PROVENANCE.md (Track 2 artifact policy, revision 2026-09-21.1)

The image contains code, static dictionaries, prompts and three JSON artifacts. It contains
**no model weights of any kind** and no data from any unit.

| Artifact | Kind | Fitted on | fit cutoffs | How the cutoff is respected |
|---|---|---|---|---|
| `src/agenthon/text/lexicon.py` | static dictionaries (hawk/dove/stress/tail vocabulary, asset terms, FX quote conventions) | hand-written; no fitting | n/a | contains no dates or outcomes |
| `prompts/*.txt`, `prompts/schemas/event_signal.json` | prompts, output schema | hand-written | n/a | — |
| `src/agenthon/resources/params_vintages.json` | a handful of scalar hyper-parameters (long-run variance weight, t-dof, drift shrink) and the (γ, λ) decision action; ≤2024 vintage adds two family-level scale/action blocks for F2/F4 | walk-forward on synthetic cards built from the published 2000-2024 practice panels; F2/F4 blocks from practice-unit outcomes reconstructed from the same panels | 2008-12-31, 2014-12-31, 2024-12-31 | runtime selects the newest vintage with fit_cutoff < card as-of; older cards use built-in priors |
| `src/agenthon/resources/policy_vintages.json` | per-action Bayesian ridge regressions (SafePolicy) on 16 context features (agenthon.policy.features.NAMES) | walk-forward full-information advantages | 2014-12-31, 2024-12-31 | same rule; only cards with outcome ≤ cutoff were used |
| `src/agenthon/resources/calibration_vintages.json` | PIT inverse-CDF knots per asset class × horizon bucket | walk-forward PITs | 2014-12-31, 2024-12-31 | same rule; **disabled** (`calibration.enabled = false`) after the M10 ablation |

Data used for fitting, selection and calibration: only the panels published in
`Agenthon-2026/track2-forecasting-public` (units/, v. cloned 2026-09-28), licensed as recorded
in each unit's `manifest.json`. No 2025+ data. No text corpus was used for fitting anything.

Models disclosed in `submission.json`: the approved House model
`nvidia/nemotron-3-super-120b-a12b` (`rl-030326-fp8`, training cutoff `unpublished`,
`access: api`). The fitted artifacts above are non-neural parameter tables of our own
numerical code; list them as `access: "local"` entries if the organiser requests it.

Library versions (image): Python 3.13-slim-bookworm, numpy 2.1.3, pandas 2.2.3,
pyarrow 18.1.0, PyYAML 6.0.2.
