import numpy as np
import pandas as pd

from hierforecast import hierarchy as H
from hierforecast import reconcile as R
from hierforecast.data import load_metadata_from_ids

from conftest import LEVELS


def _setup(toy_ids, toy_panel):
    meta = load_metadata_from_ids(pd.Series(toy_ids))
    h = H.build(meta, LEVELS)
    B, _, _ = H.bottom_matrix(toy_panel, h.bottom_ids)
    return meta, h, B


def test_bottom_up_is_the_summing_matrix_applied_to_the_bottom(toy_ids, toy_panel):
    meta, h, B = _setup(toy_ids, toy_panel)
    yhat = np.arange(h.S.shape[0] * 2, dtype=float).reshape(-1, 2)
    out = R.bottom_up(h, yhat)
    assert np.allclose(out, h.S @ yhat[h.rows_for("bottom")])
    assert R.is_coherent(h, out)[0]


def test_every_method_returns_a_coherent_forecast(toy_ids, toy_panel):
    meta, h, B = _setup(toy_ids, toy_panel)
    rng = np.random.default_rng(0)
    yhat = rng.uniform(0, 10, size=(h.S.shape[0], 3))     # deliberately incoherent
    assert not R.is_coherent(h, yhat)[0]

    props = R.historical_proportions(B)
    within = R.proportions_within(B, meta, h.bottom_ids, "store")
    outs = {
        "bu": R.bottom_up(h, yhat),
        "td": R.top_down(h, yhat, props),
        "mo": R.middle_out(h, yhat, meta, "store", within),
        "ols": R.projection(h, yhat, "ols"),
        "wls_struct": R.projection(h, yhat, "wls_struct"),
        "wls_var": R.projection(h, yhat, "wls_var",
                                residual_var=np.ones(h.S.shape[0])),
    }
    for name, out in outs.items():
        ok, err = R.is_coherent(h, out)
        assert ok, f"{name} produced an incoherent forecast, max error {err}"


def test_a_coherent_forecast_passes_through_projection_unchanged(toy_ids, toy_panel):
    """The strongest correctness property: projection must be idempotent."""
    meta, h, B = _setup(toy_ids, toy_panel)
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 5, size=(h.n_bottom, 3))
    coherent = h.S @ b                                     # coherent by construction
    for w in ("ols", "wls_struct"):
        out = R.projection(h, coherent, w)
        assert np.allclose(out, coherent, atol=1e-8), w


def test_cg_projection_matches_the_dense_formula(toy_ids, toy_panel):
    """On a hierarchy small enough to invert, check the solver against the algebra."""
    meta, h, B = _setup(toy_ids, toy_panel)
    rng = np.random.default_rng(2)
    yhat = rng.uniform(0, 10, size=(h.S.shape[0], 2))

    S = h.S.toarray()
    G = np.linalg.inv(S.T @ S) @ S.T                       # OLS, computed directly
    expected = S @ (G @ yhat)
    assert np.allclose(R.projection(h, yhat, "ols"), expected, atol=1e-8)


def test_top_down_proportions_are_a_distribution(toy_ids, toy_panel):
    meta, h, B = _setup(toy_ids, toy_panel)
    props = R.historical_proportions(B)
    assert abs(props.sum() - 1.0) < 1e-12
    assert (props >= 0).all()
    # and top down reproduces the root forecast exactly
    yhat = np.zeros((h.S.shape[0], 1)); yhat[h.rows_for("total")[0], 0] = 100.0
    out = R.top_down(h, yhat, props)
    assert abs(out[h.rows_for("total")[0], 0] - 100.0) < 1e-9


def test_within_parent_proportions_sum_to_one_per_parent(toy_ids, toy_panel):
    meta, h, B = _setup(toy_ids, toy_panel)
    within = R.proportions_within(B, meta, h.bottom_ids, "store")
    parent = meta.set_index("bottom").loc[h.bottom_ids, "store"]
    sums = pd.Series(within, index=parent.to_numpy()).groupby(level=0).sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_weighting_actually_changes_the_answer(toy_ids, toy_panel):
    """If OLS and a weighted fit agree, the weights are not being applied."""
    meta, h, B = _setup(toy_ids, toy_panel)
    rng = np.random.default_rng(3)
    yhat = rng.uniform(0, 10, size=(h.S.shape[0], 2))
    var = rng.uniform(0.5, 5.0, size=h.S.shape[0])
    a = R.projection(h, yhat, "ols")
    b = R.projection(h, yhat, "wls_var", residual_var=var)
    assert not np.allclose(a, b)
