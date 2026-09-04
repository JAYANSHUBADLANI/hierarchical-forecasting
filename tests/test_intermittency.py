import numpy as np
import pandas as pd

from hierforecast import intermittency as IM

ADI_T, CV2_T = 1.32, 0.49


def test_adi_is_periods_over_nonzero_periods():
    y = np.array([0.0, 3.0, 0.0, 0.0, 3.0])          # 5 periods, 2 sales
    adi, cv2, n = IM._stats_one(y)
    assert adi == 2.5
    assert n == 2
    assert cv2 == 0.0                                 # both sales the same size


def test_cv2_uses_only_the_nonzero_demand_sizes():
    y = np.array([0.0, 2.0, 6.0])                     # sizes 2 and 6
    adi, cv2, _ = IM._stats_one(y)
    assert adi == 1.5
    # population sd of (2, 6) is 2, mean is 4, so cv2 is (2/4)^2
    assert cv2 == 0.25


def test_classification_corners():
    assert IM.classify(1.0, 0.1, ADI_T, CV2_T) == IM.SMOOTH
    assert IM.classify(3.0, 0.1, ADI_T, CV2_T) == IM.INTERMITTENT
    assert IM.classify(1.0, 2.0, ADI_T, CV2_T) == IM.ERRATIC
    assert IM.classify(3.0, 2.0, ADI_T, CV2_T) == IM.LUMPY
    # exactly on a threshold counts as the high side, matching the stated rule
    assert IM.classify(ADI_T, 0.1, ADI_T, CV2_T) == IM.INTERMITTENT


def test_a_series_that_never_sells_is_lumpy_not_a_crash():
    adi, cv2, n = IM._stats_one(np.zeros(10))
    assert n == 0 and not np.isfinite(adi)
    assert IM.classify(adi, cv2, ADI_T, CV2_T) == IM.LUMPY


def test_padding_inflates_adi_and_the_live_window_prevents_it():
    """The point of carrying the live mask, on a case with a known answer."""
    B = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    live = np.array([[False, False, False, False, True, True, True, True]])
    ids = ["FOODS_1_001_CA_1"]

    on_live = IM.profile(B, live, ids, ADI_T, CV2_T, use_live_window=True)
    on_pad = IM.profile(B, live, ids, ADI_T, CV2_T, use_live_window=False)

    assert on_live.loc[0, "adi"] == 1.0          # sold every day it existed
    assert on_pad.loc[0, "adi"] == 2.0           # invented zeros double it
    assert on_live.loc[0, "pattern"] == IM.SMOOTH
    assert on_pad.loc[0, "pattern"] == IM.INTERMITTENT
