"""End to end on the committed sample, so the suite needs no download.

One store, two years. Small enough to run in a couple of seconds and real enough
that a change breaking the pipeline on the full panel breaks it here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierforecast import forecast as F
from hierforecast import hierarchy as H
from hierforecast import intermittency as IM
from hierforecast import pipeline as P
from hierforecast import reconcile as R
from hierforecast.config import load_config, repo_path
from hierforecast.data import load_metadata_from_ids

SAMPLE = repo_path("data/sample/sample_train_CA_1.parquet")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="run `make data` to build the sample")


@pytest.fixture(scope="module")
def sample():
    cfg = load_config()
    frame = pd.read_parquet(SAMPLE)
    frame["unique_id"] = frame["unique_id"].astype(str)
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame = frame.sort_values(["unique_id", "ds"], ignore_index=True)
    meta = load_metadata_from_ids(frame["unique_id"])
    h = H.build(meta, cfg.levels)
    B, dates, live = H.bottom_matrix(frame, h.bottom_ids)
    return cfg, frame, meta, h, B, dates, live


def test_the_sample_is_one_store_and_parses(sample):
    cfg, frame, meta, h, B, dates, live = sample
    assert meta["store_id"].nunique() == 1
    assert meta["state_id"].unique().tolist() == ["CA"]
    assert h.n_bottom == meta.shape[0] == frame.unique_id.nunique()


def test_the_sample_hierarchy_is_coherent(sample):
    cfg, frame, meta, h, B, dates, live = sample
    chk = H.check_coherence(h, B, meta, frame)
    assert chk["exact"].all()
    assert chk["all_levels_same_total"].all()


def test_every_base_method_runs_and_returns_finite_nonnegative_forecasts(sample):
    cfg, frame, meta, h, B, dates, live = sample
    Y = H.aggregate(h, B)
    fit = Y[:, :-cfg.horizon]
    for name in ["seasonal_naive", "ses", "croston_sba"]:
        out = F.METHODS[name](fit, cfg.horizon, cfg.seasonal_period)
        assert out.shape == (Y.shape[0], cfg.horizon), name
        assert np.isfinite(out).all(), name
        assert (out >= 0).all(), name


def test_reconciliation_makes_an_incoherent_forecast_coherent(sample):
    cfg, frame, meta, h, B, dates, live = sample
    Y = H.aggregate(h, B)
    fit = Y[:, :-cfg.horizon]
    yhat = F.METHODS["croston_sba"](fit, cfg.horizon, cfg.seasonal_period)
    assert not R.is_coherent(h, yhat)[0]          # Croston is not linear

    var = np.maximum(Y[:, -cfg.horizon:].var(axis=1), 1e-8)
    for recon, pred in P.reconcile_all(h, meta, yhat, fit, var).items():
        if recon == "base":
            continue
        ok, err = R.is_coherent(h, pred)
        assert ok, f"{recon} left a violation of {err}"


def test_intermittency_runs_and_classes_are_known(sample):
    cfg, frame, meta, h, B, dates, live = sample
    prof = IM.profile(B, live, h.bottom_ids,
                      cfg["intermittency"]["adi_threshold"],
                      cfg["intermittency"]["cv2_threshold"])
    assert len(prof) == h.n_bottom
    assert set(prof.pattern) <= {IM.SMOOTH, IM.ERRATIC, IM.INTERMITTENT, IM.LUMPY}
    assert prof.zero_share.between(0, 1).all()
