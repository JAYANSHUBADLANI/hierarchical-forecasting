import numpy as np

from hierforecast import forecast as F


def test_seasonal_naive_repeats_the_last_season():
    train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    out = F.seasonal_naive(train, horizon=5, season=3)
    # last season is (6, 7, 8), tiled forward
    assert list(out[0]) == [6.0, 7.0, 8.0, 6.0, 7.0]


def test_ses_on_a_constant_series_returns_that_constant():
    train = np.full((1, 30), 4.0)
    out, alpha = F.ses(train, horizon=3)
    assert np.allclose(out, 4.0)
    assert 0.0 < alpha[0] <= 1.0


def test_ses_with_alpha_one_is_the_last_value():
    train = np.array([[1.0, 5.0, 9.0]])
    out, _ = F.ses(train, horizon=2, alpha_grid=np.array([1.0]))
    assert np.allclose(out, 9.0)


def test_croston_recovers_a_regular_intermittent_rate():
    # a demand of 4 units every 4th period is a rate of 1 per period
    y = np.zeros((1, 40))
    y[0, ::4] = 4.0
    plain = F.croston(y, horizon=1, alpha=0.1, variant="croston")[0, 0]
    assert abs(plain - 1.0) < 1e-6


def test_sba_shrinks_the_croston_rate_by_the_stated_factor():
    y = np.zeros((1, 40))
    y[0, ::4] = 4.0
    plain = F.croston(y, horizon=1, alpha=0.1, variant="croston")[0, 0]
    sba = F.croston(y, horizon=1, alpha=0.1, variant="sba")[0, 0]
    assert abs(sba - plain * (1 - 0.1 / 2)) < 1e-9
    assert sba < plain


def test_a_series_that_never_sells_forecasts_zero_not_nan():
    out = F.croston(np.zeros((1, 20)), horizon=3)
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_croston_is_flat_over_the_horizon():
    y = np.zeros((1, 30)); y[0, ::5] = 2.0
    out = F.croston(y, horizon=7)
    assert len(set(out[0].round(12))) == 1
