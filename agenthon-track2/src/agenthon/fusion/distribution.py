"""L18 — distribution update engine: P1 = F(P0, Z, g, kappa), with F(P0, Z, 0, .) = P0 exactly.

Magnitude classes are mapped to horizon-sd multiples through the unit's OWN empirical |z|
quantiles (TailProfile.abs_z_quantiles; "strong" ~ the 90th percentile of historical standardised
moves), never through an LLM-supplied number (L14).

Mechanism A (tilt, default)
    mu_1    = mu_0 + g * alpha * (2 p_up - 1) * m(class) * f_pers(s) * sd_0
    sigma_1 = sigma_0 * exp(g * beta * v(vol_view) + lambda_kappa * kappa * sqrt(g))
Multiplicative scale => sigma_1 > 0 always (Proposition 1). Narrowing (v < 0) is only honoured for
an *established* driver (Playbook: inferred adjustments may move the centre, never narrow).

Mechanism B (mixture, genuine branching or kappa > kappa*)
    P1 = (1 - g) P0 + g * sum_s p_s P_s, each P_s a shifted/scaled copy of P0; a valid density
    (Proposition 2), P1 = P0 at g = 0.

Event-triggered tail (L20, F4-weighted): a small-weight wide component on the side the text
flags. Targets the 1%/99% pinball points directly instead of widening everything.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..families.base import FamilyPolicy
from ..types import GateDecision, JointBelief, Scenario, SemanticSignal

_VOL = {"lower": -0.5, "same": 0.0, "higher": 1.0, "much_higher": 2.0}


def _persist(kind: str, s: int) -> float:
    ref = {"transient": 21.0, "medium": 63.0}.get(kind)
    return 1.0 if ref is None else math.sqrt(min(1.0, ref / max(s, 1)))


def distribution_update(p0: JointBelief, sig: SemanticSignal, gate: GateDecision,
                        absz: dict[str, float], policy: FamilyPolicy,
                        cfg: dict[str, Any]) -> tuple[JointBelief, list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    g = float(gate.g)
    if g <= 0.0:
        return p0, ledger
    uc = cfg["update"]
    grid = p0.grid
    sd = p0.cell_sd()
    out = p0.copy()
    alpha = float(uc["alpha_shift"]) * policy.text_strength
    beta = float(uc["beta_vol"])
    lamk = float(uc["lambda_kappa"])
    narrow_ok = bool(uc.get("allow_text_narrowing", False)) or sig.established
    shift = np.zeros(grid.d)
    logscale = np.zeros(grid.d)
    for i, c in enumerate(grid.cells):
        a = sig.assets.get(c.asset)
        if a is None:
            continue
        m = absz.get(a.magnitude, 0.0) if a.magnitude != "none" else 0.0
        shift[i] = g * alpha * (2 * a.p_up - 1) * m * _persist(sig.persistence, c.steps) * sd[i]
        v = _VOL.get(a.vol_view, 0.0)
        if v < 0 and not narrow_ok:
            v = 0.0
        logscale[i] = g * beta * v + lamk * gate.kappa * math.sqrt(g)
    scale = np.exp(np.minimum(logscale, np.log(float(uc.get("max_text_vol_mult", 2.0)))))
    if gate.mechanism == "mixture" and sig.scenarios:
        base = out.scenarios[0]
        comps = [Scenario("numeric_prior", 1.0 - g, base.shift.copy(), base.scale * scale,
                          "text-blind path")]
        for sv in sig.scenarios:
            sh = np.zeros(grid.d)
            scv = np.ones(grid.d)
            for i, c in enumerate(grid.cells):
                d = sv.direction.get(c.asset, 0)
                sh[i] = d * absz.get(sv.magnitude, 1.0) * _persist(sig.persistence, c.steps) * sd[i] \
                    * min(1.0, 1.5 * alpha)
                scv[i] = math.exp(beta * max(_VOL.get(sv.vol_view, 0.0), 0.0 if not narrow_ok else -0.5))
            scv = np.minimum(scv, float(uc.get("max_text_vol_mult", 2.0)))
            comps.append(Scenario(sv.name, g * sv.probability, sh, scv * scale, "text scenario"))
            ledger.append({"step": "scenario", "name": sv.name, "weight": round(g * sv.probability, 4),
                           "magnitude": sv.magnitude, "direction": sv.direction})
        out.scenarios = comps
    else:
        out.mu = p0.mu + shift
        out.scenarios = [Scenario(s.name, s.weight, s.shift, s.scale * scale, s.note) for s in out.scenarios]
    # event-triggered one-sided tail component
    sides = {a: s.tail_side for a, s in sig.assets.items() if s.tail_side != "none"}
    if sides:
        strong = sig.established or any(s.magnitude in ("strong", "extreme") for s in sig.assets.values())
        wt = g * float(uc["tail_weight_max"]) * policy.tail_strength * (1.0 if strong else 0.5)
        wt = float(min(wt, 0.25))
        if wt > 1e-4:
            sh = np.zeros(grid.d)
            for i, c in enumerate(grid.cells):
                side = sides.get(c.asset, "none")
                sh[i] = {"up": 1.0, "down": -1.0}.get(side, 0.0) * absz.get("strong", 1.645) * sd[i]
            tsc = np.full(grid.d, float(uc["tail_scale"]))
            for s in out.scenarios:
                s.weight *= (1.0 - wt)
            out.scenarios.append(Scenario("tail_event", wt, sh, tsc * scale, "event-triggered tail"))
            ledger.append({"step": "tail", "weight": round(wt, 4), "sides": sides})
    ledger.append({"step": "tilt" if gate.mechanism != "mixture" else "mixture",
                   "gate": round(g, 4), "kappa": round(gate.kappa, 4),
                   "max_shift_sd": round(float(np.max(np.abs(shift) / np.maximum(sd, 1e-12))), 4),
                   "vol_mult_range": [round(float(scale.min()), 4), round(float(scale.max()), 4)]})
    return out, ledger
