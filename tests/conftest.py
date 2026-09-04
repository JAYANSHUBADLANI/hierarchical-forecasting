"""A tiny hierarchy whose every number is checkable by hand.

Two stores in one state plus two stores in another, two departments in one
category, four bottom series. Small enough that the summing matrix, the
aggregates and later the reconciled forecasts can all be written out by hand and
compared, which is the only way to know the linear algebra is right rather than
merely self consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

LEVELS = ["total", "state", "store", "store_cat", "store_dept", "bottom"]


@pytest.fixture
def toy_ids() -> list[str]:
    return ["FOODS_1_001_CA_1", "FOODS_2_002_CA_1",
            "FOODS_1_001_TX_1", "FOODS_1_001_TX_2"]


@pytest.fixture
def toy_panel(toy_ids) -> pd.DataFrame:
    dates = pd.date_range("2011-01-01", periods=4, freq="D")
    values = {
        "FOODS_1_001_CA_1": [1, 0, 2, 3],
        "FOODS_2_002_CA_1": [0, 0, 5, 1],
        "FOODS_1_001_TX_1": [4, 4, 0, 0],
        "FOODS_1_001_TX_2": [2, 2, 2, 2],
    }
    rows = [{"unique_id": k, "ds": d, "y": float(v)}
            for k, vs in values.items() for d, v in zip(dates, vs)]
    return pd.DataFrame(rows).sort_values(["unique_id", "ds"], ignore_index=True)
