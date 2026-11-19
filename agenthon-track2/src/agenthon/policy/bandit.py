"""Contextual policy with safe (pessimistic) improvement over a default action.

Why not PPO: financial rewards are dominated by outcome noise (Var(Y|X,T) >> text signal), and in
offline walk-forward we observe y, so the reward of EVERY action is computable (full information).
The problem is then cost-sensitive contextual decision-making, not exploration. We fit, per action
a, a Bayesian ridge model of the ADVANTAGE over the default action,
    r_a(x) = S_default(y) - S_a(y)      (variance-controlled: the card's own difficulty cancels),
with posterior N(theta_a, sigma^2 A_a^{-1}). At inference we choose
    a* = argmax_a  mean_a(x) - k * sd_a(x),   and keep the default unless that LCB > 0
(SPIBB-style safe policy improvement). `update` also supports online LinUCB/Thompson learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LinearArm:
    dim: int
    lam: float = 10.0
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n: int = 0
    sse: float = 0.0

    def __post_init__(self) -> None:
        self.A = self.lam * np.eye(self.dim)
        self.b = np.zeros(self.dim)

    def update(self, X: np.ndarray, r: np.ndarray) -> None:
        X = np.atleast_2d(X)
        r = np.atleast_1d(r)
        self.A += X.T @ X
        self.b += X.T @ r
        self.n += len(r)
        th = self.theta
        self.sse += float(np.sum((r - X @ th) ** 2))

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def noise_var(self) -> float:
        return self.sse / max(self.n - self.dim, 1) if self.n > self.dim else 1.0

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        th = self.theta
        v = float(x @ np.linalg.solve(self.A, x)) * self.noise_var()
        return float(x @ th), float(np.sqrt(max(v, 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {"dim": self.dim, "lam": self.lam, "A": self.A.tolist(), "b": self.b.tolist(),
                "n": self.n, "sse": self.sse}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LinearArm":
        arm = cls(int(d["dim"]), float(d.get("lam", 10.0)))
        arm.A = np.asarray(d["A"], dtype=float)
        arm.b = np.asarray(d["b"], dtype=float)
        arm.n = int(d.get("n", 0))
        arm.sse = float(d.get("sse", 0.0))
        return arm


class SafePolicy:
    def __init__(self, n_actions: int, dim: int, default: int, lam: float = 10.0):
        self.arms = [LinearArm(dim, lam) for _ in range(n_actions)]
        self.default = default

    def fit(self, X: np.ndarray, adv: np.ndarray) -> None:
        """X [n, dim]; adv [n, n_actions] advantages over the default action (default column 0)."""
        for a, arm in enumerate(self.arms):
            if a == self.default:
                continue
            ok = np.isfinite(adv[:, a])
            arm.update(X[ok], adv[ok, a])

    def select(self, x: np.ndarray, pessimism: float = 1.0) -> tuple[int, dict[str, float]]:
        best, bi, info = 0.0, self.default, {}
        for a, arm in enumerate(self.arms):
            if a == self.default or arm.n == 0:
                continue
            m, s = arm.predict(x)
            lcb = m - pessimism * s
            info[str(a)] = lcb
            if lcb > best:
                best, bi = lcb, a
        return bi, info

    def thompson(self, x: np.ndarray, rng: np.random.Generator) -> int:
        vals = []
        for a, arm in enumerate(self.arms):
            if a == self.default:
                vals.append(0.0)
                continue
            m, s = arm.predict(x)
            vals.append(m + s * rng.standard_normal())
        return int(np.argmax(vals))

    def to_dict(self) -> dict[str, Any]:
        return {"default": self.default, "arms": [a.to_dict() for a in self.arms]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SafePolicy":
        arms = [LinearArm.from_dict(x) for x in d["arms"]]
        p = cls(len(arms), arms[0].dim, int(d["default"]))
        p.arms = arms
        return p
