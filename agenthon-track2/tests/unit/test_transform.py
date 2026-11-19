import numpy as np
import pandas as pd

from agenthon.numerical.transform import counted_steps, level_steps, return_steps
from agenthon.timeutil import intervals_days


def test_gap_rule_drops_withheld_decade():
    early = pd.bdate_range("1995-01-02", periods=500)
    idx = early.append(pd.DatetimeIndex([pd.Timestamp("2015-07-31")]))
    s = pd.Series(np.r_[np.full(500, 8.28), 6.21], index=idx)
    st = level_steps(s)
    assert len(st) == 499  # the step across the hole is dropped, not a -2.07 one-day move
    assert np.allclose(st.to_numpy(), 0.0)


def test_return_steps_are_log1p_and_skip_row0():
    idx = pd.bdate_range("2020-01-01", periods=4)
    s = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    st = return_steps(s)
    assert len(st) == 3 and np.allclose(st.to_numpy(), np.log1p([0.02, -0.01, 0.03]))


def test_intervals_independent_of_resolution():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-03"]).as_unit("us")
    assert intervals_days(idx)[0] == 2.0
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-03"]).as_unit("s")
    assert intervals_days(idx)[0] == 2.0


def test_m0_horizon_override_rule():
    daily = pd.bdate_range("2024-01-01", periods=129)
    # exemplar: counted 24 vs declared 21 -> ratio < 2 -> declared kept
    assert counted_steps(daily, daily[-1], pd.Timestamp("2024-07-31"), 21) in (21,)
    monthly = pd.date_range("2000-01-01", periods=280, freq="MS")
    # monthly panel, declared 140 BD, target 8 months after last obs -> counted wins
    assert counted_steps(monthly, monthly[-1], monthly[-1] + pd.DateOffset(months=8), 140) == 8
    assert counted_steps(monthly, monthly[-1], None, 140) == 140
