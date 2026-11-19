"""Typed contracts passed between layers L0..L23.

Every layer consumes and produces one of these objects and nothing else. That is what makes the
fail-safe design possible: a layer that fails can be replaced by its documented neutral element
(e.g. an empty `TextEvidence` forces the information gate to zero, which by construction returns
the numerical prior unchanged).

Shape conventions
-----------------
* A *cell* is one ``(asset, horizon)`` pair. Cells are always enumerated in **M0 order**: sorted by
  asset id, then by horizon ascending (docs/M0-BASELINE.md section 3.9). Keeping one canonical order
  everywhere means the draw matrix ``[n_draws, n_cells]`` lines up with the official baseline.
* Draw tensors are ``float64`` arrays of shape ``[n_draws, n_cells]``. A draw index is one coherent
  joint scenario across all cells -- never independent per-cell sampling.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = ("T2-F1", "T2-F2", "T2-F3", "T2-F4")


# --------------------------------------------------------------------------------------------
# L0 — the card
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Card:
    unit_id: str
    asof: str  # trusted cutoff = [provenance] data_cutoff (what forecast_meta.json must echo)
    family: str  # one of FAMILIES, or "T2-UNKNOWN"
    assets: tuple[str, ...]  # as listed in [targets] asset_ids (card order)
    horizons: tuple[int, ...]  # declared horizon keys (card order)
    target_type: str  # "level" | "log_return"
    target_frequency: str  # "daily" | "monthly"
    value_unit: str
    n_draws_min: int
    weights: tuple[float, float, float]  # (marginal, joint, tail) as declared by the card
    tail_levels: tuple[float, ...]
    joint: str  # "variogram" | "energy"
    title: str = ""
    description: str = ""
    text_notes: str = ""
    prose: str = ""  # forecast_card.md, if present (a legitimate /input file)
    observation_periods: tuple[str, ...] | None = None  # monthly targets, one per grid cell
    target_dates: tuple[str, ...] | None = None  # rarely present (exemplar only)
    card_dir: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_cells(self) -> int:
        return len(self.assets) * len(self.horizons)

    @property
    def is_single_cell(self) -> bool:
        return self.n_cells == 1

    def effective_weights(self) -> tuple[float, float, float]:
        """Scorer's weights after the single-cell renormalisation (scoring.py, 2026-08-24 ruling)."""
        wm, wj, wt = self.weights
        if self.is_single_cell and self.joint == "variogram":
            live = wm + wt
            return (wm / live, 0.0, wt / live)
        return (wm, wj, wt)


@dataclass(frozen=True)
class Cell:
    asset: str
    horizon: int  # declared key, written verbatim to forecast.parquet
    steps: int  # panel steps used by the walk (M0 section 3.7 conversion)


@dataclass(frozen=True)
class Grid:
    cells: tuple[Cell, ...]  # M0 order: sorted by asset, then horizon

    @property
    def assets(self) -> list[str]:
        return sorted({c.asset for c in self.cells})

    def index_of(self, asset: str, horizon: int) -> int:
        for i, c in enumerate(self.cells):
            if c.asset == asset and c.horizon == horizon:
                return i
        raise KeyError((asset, horizon))

    def cells_of(self, asset: str) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c.asset == asset]

    @property
    def d(self) -> int:
        return len(self.cells)


# --------------------------------------------------------------------------------------------
# L1 — point-in-time data
# --------------------------------------------------------------------------------------------
@dataclass
class AssetHistory:
    asset: str
    panel_id: str
    series: pd.Series  # float values indexed by pd.DatetimeIndex, every date <= asof, sorted
    is_transfer: bool = False  # early window + as-of row with a withheld gap (F2 transfer)
    asset_class: str = "other"  # rates | fx | factor | macro | em_fx | other


@dataclass
class TextDoc:
    doc_id: str
    timestamp: str  # availability timestamp (release date), ISO date
    doc_type: str
    source: str
    institution: str
    text: str
    path: str = ""


@dataclass
class PointInTimeData:
    asof: str
    histories: dict[str, AssetHistory]  # every asset in any panel (targets and context)
    docs: list[TextDoc]
    dropped_rows: int = 0  # defence-in-depth counters (should be 0: staging already truncates)
    dropped_docs: int = 0


# --------------------------------------------------------------------------------------------
# L2-L4 — numerical state
# --------------------------------------------------------------------------------------------
@dataclass
class AssetState:
    asset: str
    asset_class: str
    anchor: float  # level: last observation; log_return: 0.0
    step_sd_m0: float  # sd of the trailing-300 steps (what M0 uses)
    step_sd_ewma: float  # current conditional step sd (EWMA)
    step_sd_long: float  # long-run step sd (robust, multi-year)
    drift_m0: float  # mean step over the trailing-300 window (M0's mu)
    drift_long: float  # long-run mean step (all usable history)
    momentum_z: float  # recent trend as a z-score (sign used by the contradiction detector)
    vol_ratio: float  # step_sd_ewma / step_sd_long
    variance_ratio: dict[int, float]  # empirical VR(s) (shrunk) for the requested step counts
    n_steps: int  # usable steps in the full history
    last_date: str
    level_floor: float | None = None  # e.g. an effective lower bound for yields
    extras: dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeBelief:
    labels: tuple[str, ...]  # e.g. ("calm", "normal", "stressed")
    probs: np.ndarray  # [K], sums to 1
    source: str = "numeric"  # "numeric" | "numeric+text" | "global"

    @property
    def stressed(self) -> float:
        return float(self.probs[-1]) if len(self.probs) else 0.0


@dataclass
class NumericalState:
    card: Card
    grid: Grid
    assets: dict[str, AssetState]
    step_corr: np.ndarray  # [n_assets, n_assets] correlation of steps (grid.assets order)
    regime: RegimeBelief
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# L5-L7 / L18-L21 — the distribution (parametric belief before sampling)
# --------------------------------------------------------------------------------------------
@dataclass
class Scenario:
    """One mixture component. Shifts/scales are per cell; `weight` is its mixture probability."""

    name: str
    weight: float
    shift: np.ndarray  # [d] additive shift of the centre (in target units)
    scale: np.ndarray  # [d] multiplicative scale on the base spread (> 0)
    note: str = ""


@dataclass
class JointBelief:
    """A structured joint predictive distribution over the grid.

    base: path-coherent Gaussian scale-mixture with centre `mu`, per-asset step variance term
    structure `var_path[asset][k]` at the grid's sorted unique step counts, and increment
    correlation `corr` across assets. Scenarios multiply/shift the base; every draw selects one
    scenario, applied to all cells (a coherent joint story, never a per-cell coin flip).
    """

    grid: Grid
    mu: np.ndarray  # [d]
    step_points: tuple[int, ...]  # sorted unique step counts across the grid
    var_path: dict[str, np.ndarray]  # asset -> cumulative variance at each step point (monotone)
    corr: np.ndarray  # [n_assets, n_assets] increment correlation (grid.assets order)
    df: float  # scale-mixture degrees of freedom (inf -> Gaussian)
    scenarios: list[Scenario]
    floors: dict[str, float] = field(default_factory=dict)  # asset -> soft lower bound
    provenance: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "JointBelief":
        return dataclasses.replace(
            self,
            mu=self.mu.copy(),
            var_path={k: v.copy() for k, v in self.var_path.items()},
            corr=self.corr.copy(),
            scenarios=[dataclasses.replace(s, shift=s.shift.copy(), scale=s.scale.copy())
                       for s in self.scenarios],
            floors=dict(self.floors),
            provenance=dict(self.provenance),
        )

    def cell_sd(self) -> np.ndarray:
        """Gaussian-base marginal sd per cell (before mixture/scenario effects)."""
        out = np.empty(self.grid.d)
        for i, c in enumerate(self.grid.cells):
            k = self.step_points.index(c.steps)
            out[i] = float(np.sqrt(max(self.var_path[c.asset][k], 0.0)))
        return out


# --------------------------------------------------------------------------------------------
# L8-L14 — text evidence and the structured semantic signal
# --------------------------------------------------------------------------------------------
@dataclass
class Event:
    event_id: str
    doc_id: str
    timestamp: str
    doc_type: str
    institution: str
    text: str  # a compact passage (not the whole document)
    topics: tuple[str, ...]
    assets: tuple[str, ...]  # target/context assets this passage plausibly touches
    hawk: float  # lexicon tone in [-1, 1]; + hawkish / tightening
    stress: float  # lexicon stress/uncertainty intensity in [0, 1]
    tail_terms: int  # count of explicit tail/shock vocabulary
    cluster: int = -1  # dedup cluster id
    corroboration: int = 1  # documents merged into this cluster
    novelty: float = 1.0
    score: float = 0.0  # retrieval score (eq. 16)


@dataclass
class AssetSignal:
    """Semantic view for one asset (the LLM's job: T -> Z; never a number for the market)."""

    asset: str
    p_up: float  # probability the level ends higher than the random-walk centre
    p_down: float
    magnitude: str  # "none" | "weak" | "moderate" | "strong"
    vol_view: str  # "lower" | "same" | "higher" | "much_higher"
    tail_side: str  # "none" | "up" | "down" | "both"


@dataclass
class ScenarioView:
    name: str
    probability: float
    direction: dict[str, int]  # asset -> -1/0/+1
    magnitude: str  # weak | moderate | strong | extreme
    vol_view: str


@dataclass
class SemanticSignal:
    """Z_t: low-dimensional, auditable, schema-validated."""

    assets: dict[str, AssetSignal]
    scenarios: list[ScenarioView]
    persistence: str  # transient | medium | persistent
    regime_evidence: str  # supports_calm | neutral | supports_stress
    confidence: float  # [0, 1] evidence consistency
    novelty: float  # [0, 1]
    established: bool  # the key driver is a stated fact (decision taken / print released)
    source: str  # "llm" | "lexicon" | "neutral"
    cited_docs: tuple[str, ...] = ()
    summary: str = ""

    @staticmethod
    def neutral(assets: list[str], source: str = "neutral") -> "SemanticSignal":
        return SemanticSignal(
            assets={a: AssetSignal(a, 0.5, 0.5, "none", "same", "none") for a in assets},
            scenarios=[],
            persistence="transient",
            regime_evidence="neutral",
            confidence=0.0,
            novelty=0.0,
            established=False,
            source=source,
        )


@dataclass
class TextEvidence:
    events: list[Event]
    retrieved: list[Event]
    signal: SemanticSignal
    retrieval_quality: float  # max normalised retrieval score in [0, 1]
    corroboration_max: int
    llm_calls: int = 0
    llm_ok: bool = False
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def empty(assets: list[str], note: str = "no text") -> "TextEvidence":
        return TextEvidence([], [], SemanticSignal.neutral(assets), 0.0, 0, notes=[note])


# --------------------------------------------------------------------------------------------
# L15-L16 / policy
# --------------------------------------------------------------------------------------------
@dataclass
class GateDecision:
    g: float  # information gate in [0, 1]; g == 0 => P1 == P0 exactly
    kappa: float  # contradiction in [0, 1]
    mechanism: str  # "none" | "tilt" | "mixture"
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyAction:
    gamma: float  # score-aware reweighting exponent (0 = submit the honest belief)
    text_strength: float  # multiplier on the gate (0 disables text updates)
    lam: float = 0.0  # fraction of draws from M0's own law
    source: str = "default"
    diagnostics: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
@dataclass
class ForecastResult:
    card: Card
    grid: Grid
    draws: np.ndarray  # [n_draws, d]
    path: str  # "full" | "numeric_only" | "failsafe_m0" | "failsafe_minimal"
    ledger: list[dict[str, Any]] = field(default_factory=list)  # rationale adjustment ledger
    diagnostics: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
