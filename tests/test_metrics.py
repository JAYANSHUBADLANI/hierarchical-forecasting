import numpy as np

from hierforecast import metrics as M


def test_scale_is_the_mean_absolute_seasonal_difference():
    train = np.array([[1.0, 2.0, 4.0, 8.0]])          # season 2 diffs: 3, 6
    assert M.seasonal_naive_scale(train, season=2)[0] == 4.5


def test_mase_of_a_perfect_forecast_is_zero():
    train = np.array([[1.0, 2.0, 4.0, 8.0]])
    scale = M.seasonal_naive_scale(train, season=2)
    actual = np.array([[3.0, 3.0]])
    assert M.mase(actual, actual.copy(), scale)[0] == 0.0


def test_mase_is_error_over_scale():
    train = np.array([[1.0, 2.0, 4.0, 8.0]])          # scale 4.5
    scale = M.seasonal_naive_scale(train, season=2)
    actual = np.array([[10.0, 10.0]])
    pred = np.array([[1.0, 10.0]])                    # errors 9 and 0, mean 4.5
    assert M.mase(actual, pred, scale)[0] == 1.0


def test_short_history_gives_nan_not_infinity():
    train = np.array([[1.0, 2.0]])
    assert np.isnan(M.seasonal_naive_scale(train, season=7)[0])
    assert np.isnan(M.rmsse(np.array([[1.0]]), np.array([[1.0]]), train, 7)[0])


def test_scale_ignores_days_the_series_was_not_live():
    # the padded zeros would halve the scale if they were counted
    train = np.array([[0.0, 0.0, 5.0, 9.0]])
    live = np.array([[False, False, True, True]])
    with_pad = M.seasonal_naive_scale(train, season=2, live=None)[0]
    live_only = M.seasonal_naive_scale(train, season=2, live=live)[0]
    assert with_pad == 7.0        # (|5-0| + |9-0|) / 2
    assert np.isnan(live_only)    # no pair of days where both ends are live
