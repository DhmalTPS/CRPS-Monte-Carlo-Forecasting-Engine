# AGENTS.md — rules for anyone (human or agent) changing this repository

## Non-negotiables
1. **Never cost a card.** Every code path ends in `output.write_forecast` or
   `safety.fail_safe_forecast`. New stages must degrade to their neutral element
   (empty evidence ⇒ g = 0 ⇒ P1 ≡ P0) instead of raising past `main.run`.
2. **`/output` holds exactly three files.** Logs go to stderr. No temp files survive.
3. **Point in time.** Nothing dated after the card's as-of may influence a forecast,
   including via normalisation, retrieval, calibration or a fitted parameter. Fitted
   artifacts go in `src/agenthon/resources/*_vintages.json` with a `fit_cutoff`; a card may
   only use a vintage whose fit_cutoff < its as-of.
4. **No cross-unit lookup.** Never bake a value derived from any unit's target into the
   image. Practice outcomes are for evaluation and permitted tuning of low-dimensional
   family scalars only, and only in vintages dated after them.
5. **House model only**, through `MODEL_ENDPOINT` + `MODEL_TOKEN`; no vendor APIs, no tools,
   temperature/seed pinned, thinking off for structured calls. No neural checkpoints in the
   image without written organizer approval.
6. **Text may widen freely but narrow only on an established fact** (`allow_text_narrowing`).

## Changing the model
* Research lives in `offline/`; production in `src/`. The lab calls the production code
  (`agenthon.main.forecast_from_pit`) — do not fork formulas into `offline/`.
* A change ships only if the walk-forward (`offline/walk_forward/run.py --overrides
  '{"_vintaged": true}'`) improves the out-of-sample slices with the quarter-block CI
  (`offline/walk_forward/aggregate.py`), and does not worsen `stress.py`.
* The ablation ladder is fixed (`offline/ablations/_ladder.py`); do not reorder rungs after
  seeing results.
* After editing `agenthon.config.DEFAULTS`, run `python scripts/sync_config.py`
  (`tests/unit/test_config.py` fails if `configs/default.yaml` drifts — a stale YAML
  silently overrides the code; this happened once).

## Before any upload
```
python -m pytest
python scripts/run_local.py --units <units> --no-llm --jobs 6
python scripts/validate_submission.py --units <units> --outputs artifacts/forecasts   # 104/104
docker build --platform linux/amd64 -t agenthon-t2:<ver> . && bash submission/run.sh <unit>
```
Development uploads are limited (5/day, 20 total); local validation costs none.
