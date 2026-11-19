"""Helpers shared by tests (importable because pytest puts tests/ on sys.path via rootdir conftest)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import write_unit  # noqa: E402


def write_transfer_card(root: pathlib.Path) -> pathlib.Path:
    u = write_unit(root, uid="t2-F2-transfer-2015", fam="T2-F2", assets=("CNY",), horizons=(64,),
                   asof="2015-07-31", docs=False, kind="fx")
    (u / "forecast_card.md").write_text("Target level on the as-of date (CNY = 6.2097, unit cny_per_usd).",
                                        encoding="utf-8")
    return u
