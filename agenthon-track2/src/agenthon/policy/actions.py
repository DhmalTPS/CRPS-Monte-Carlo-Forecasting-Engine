"""Discrete action set of the adaptive policy (what the policy is allowed to decide).

The statistical backbone stays fixed; the policy decides how far the submission moves from the
honest belief toward the metric-optimal act:
  gamma   score-aware reweighting exponent (resample the belief with weights (1/S_M0(y))^gamma)
  lambda  fraction of draws from the baseline's own law (model averaging; caps blow-ups)
Text strength is NOT a policy action: the historical walk-forward has no corpus, so there are no
data to learn it from; it stays with the family policy and the information gate.
Default action = config (decision.gamma, decision.lambda): the safe baseline the policy must beat.
"""

from __future__ import annotations

GAMMAS = (0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0)
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4)
ACTIONS = tuple((g, l) for g in GAMMAS for l in LAMBDAS)


def action_key(gamma: float, lam: float) -> str:
    return f"g{gamma:.2f}" if lam == 0 else f"g{gamma:.2f}_l{lam:.2f}"


def action_index(gamma: float, lam: float) -> int:
    best, bi = 1e9, 0
    for i, (g, l) in enumerate(ACTIONS):
        d = abs(g - gamma) + abs(l - lam)
        if d < best:
            best, bi = d, i
    return bi
