"""Resolution-independent date arithmetic.

pandas >= 2 no longer guarantees nanosecond datetime resolution (pandas 3 defaults to
microseconds for many constructors), so `DatetimeIndex.asi8 / 86_400e9` silently yields wrong day
counts. Every interval in this package goes through these helpers instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def day_numbers(index) -> np.ndarray:
    """Days since the epoch as float64, whatever the index's datetime resolution."""
    arr = pd.DatetimeIndex(index).to_numpy().astype("datetime64[s]").astype(np.int64)
    return arr / 86400.0


def intervals_days(index) -> np.ndarray:
    """Gaps between consecutive dates, in days."""
    return np.diff(day_numbers(index))
