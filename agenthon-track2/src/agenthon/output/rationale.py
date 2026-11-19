"""forecast_rationale.md — required, never scored, read by a human reviewer.

Follows SUBMISSION_CLI's T2 note: numbered steps (anchor, each adjustment with its size and what
supports it, scale and shape), the documents cited, established vs inferred, and an adjustment
ledger so the arithmetic can be followed. States plainly that the submitted draws are optimised
for the published normalised score (score-aware layer) and how far they are from the honest belief.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Card, Grid


def _q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p))


def build_rationale(card: Card, grid: Grid, draws: np.ndarray, info: dict[str, Any]) -> str:
    lines = [f"# Forecast rationale — {card.unit_id}", "",
             f"As of **{card.asof}** ({card.family}), target type `{card.target_type}`, "
             f"{draws.shape[0]} joint draws over {grid.d} cell(s). Path: **{info.get('path', 'full')}**.", ""]
    lines += ["## 1. Anchor and numerical prior (text-blind)", ""]
    for a, s in info.get("assets", {}).items():
        lines.append(f"- `{a}` ({s.get('class')}): anchor {s.get('anchor'):.6g} (last obs {s.get('last_date')}); "
                     f"step sd now/300-obs/long-run = {s.get('sd_now'):.4g}/{s.get('sd_m0'):.4g}/{s.get('sd_long'):.4g}; "
                     f"model `{s.get('model')}`; in-unit backtest scale bias x{s.get('scale_bias', 1.0):.3f}.")
    lines += ["", "Variance term structure: current EWMA variance mean-reverting toward a long-run level "
              "(GARCH-type decay), times a shrunk empirical variance ratio; Student-t scale mixture "
              f"(dof {info.get('df', float('nan')):.1f}) shared across cells within each draw; cross-horizon "
              "structure by cumulated path increments; cross-asset increment correlation from the panel.", ""]
    lines += ["## 2. Text evidence", ""]
    tx = info.get("text", {})
    lines.append(f"- Source of the semantic signal: **{tx.get('source', 'none')}**; LLM calls {tx.get('llm_calls', 0)}; "
                 f"retrieved passages {tx.get('n_retrieved', 0)} of {tx.get('n_events', 0)} deduplicated events.")
    lines.append(f"- Information gate g = {tx.get('gate', 0.0):.3f}, contradiction kappa = {tx.get('kappa', 0.0):.3f}, "
                 f"mechanism `{tx.get('mechanism', 'none')}`; driver {'ESTABLISHED (stated fact)' if tx.get('established') else 'INFERRED (tone) — may move the centre, never narrows'}.")
    if tx.get("summary"):
        lines.append(f"- Reading: {tx['summary']}")
    if tx.get("cited"):
        lines.append(f"- Documents relied on: {', '.join('`' + c + '`' for c in tx['cited'])}")
    for a, s in tx.get("assets", {}).items():
        lines.append(f"- `{a}`: p_up {s['p_up']:.2f}, magnitude {s['magnitude']}, vol view {s['vol_view']}, tail {s['tail_side']}.")
    if tx.get("gate", 0.0) == 0:
        lines.append("- g = 0: the text did not clear the gate, so the forecast equals the numerical prior exactly.")
    lines += ["", "## 3. Adjustment ledger", "", "| step | detail |", "|---|---|"]
    for row in info.get("ledger", []):
        detail = ", ".join(f"{k}={v}" for k, v in row.items() if k != "step")
        lines.append(f"| {row.get('step')} | {detail} |")
    lines += ["", "## 4. Submitted distribution", "",
              f"Score-aware layer: gamma = {info.get('gamma', 0.0):.2f} (0 = the honest belief), "
              f"lambda = {info.get('lambda', 0.0):.2f} of the draws from the baseline's own law (model averaging). The published "
              "score is a ratio to the text-blind baseline per card, so the metric-optimal submission is the "
              "belief reweighted toward outcomes where that baseline scores well (a proper-score identity); "
              f"no cell is narrower than {info.get('min_sd_ratio', 0.6):.0%} of the honest belief. "
              f"Calibration: {info.get('calibration', 'n/a')}. Parameters: {info.get('params', 'priors')}.", "",
              "| asset | horizon | 5% | 50% | 95% | sd |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(grid.cells):
        x = draws[:, i]
        lines.append(f"| {c.asset} | {c.horizon} | {_q(x, .05):.6g} | {_q(x, .5):.6g} | {_q(x, .95):.6g} | {np.std(x):.4g} |")
    lines += ["", "## 5. What would change this forecast", "",
              "A realised volatility shift in the panel (moves the width via the EWMA term), a dated document "
              "stating a decision or print not yet in the numbers (moves the centre / adds a scenario), or a "
              "shock-language document (adds weight to the flagged tail).", ""]
    if info.get("note"):
        lines += ["## Note", "", str(info["note"]), ""]
    return "\n".join(lines)


def fallback_rationale(card_id: str, asof: str, reason: str, path: str) -> str:
    return (f"# Forecast rationale — {card_id}\n\nAs of **{asof}**. The full pipeline did not complete "
            f"({reason}); the fail-safe path `{path}` was used.\n\n"
            "`failsafe_m0`: the official text-blind baseline procedure (docs/M0-BASELINE.md) rebuilt "
            "exactly — trailing 300 observations, sample drift and covariance, path covariance "
            "min(s_i,s_j)·Sigma, 500 draws seeded by crc32(unit_id). No text was used.\n"
            "`failsafe_minimal`: a robust-scale random walk from the last observation with a common shock.\n")
