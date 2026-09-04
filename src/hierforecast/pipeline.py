"""The study, end to end.

Two forecast origins, and they do different jobs.

The **validation origin** fits on everything except the last 28 days of the
training file and forecasts those 28 days. Its errors are what the reconciliation
weights are estimated from, and its history is what the top down and middle out
proportions come from. Nothing here ever sees the evaluation window.

The **test origin** fits on the whole training file and forecasts the 28 days of
the held out test file. This is the only thing that gets scored.

Keeping those apart is the difference between a measurement and a number that
looks good.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import forecast as F
from . import gbm as G
from . import hierarchy as H
from . import metrics as M
from . import reconcile as R
from .config import Config
from .hierarchy import Hierarchy

BASE_METHODS = ["seasonal_naive", "ses", "croston_sba", "gbm"]
RECON_METHODS = ["base", "bottom_up", "top_down", "middle_out",
                 "ols", "wls_struct", "wls_var"]
MIDDLE_LEVEL = "store"


@dataclass
class Origin:
    """One forecast origin: what was fitted on, and what came out."""
    name: str
    Y_fit: np.ndarray
    dates_fit: pd.DatetimeIndex
    Y_actual: np.ndarray
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def make_base_forecasts(origin: Origin, h: Hierarchy, cfg: Config) -> None:
    level_code = pd.Categorical(h.nodes["level"], categories=cfg.levels).codes
    bottom_mask = (h.nodes["level"] == "bottom").to_numpy()
    for name in BASE_METHODS:
        t0 = time.perf_counter()
        if name == "gbm":
            preds, _ = G.fit_predict(origin.Y_fit, origin.dates_fit, cfg.horizon,
                                     level_code, cfg.seed, bottom_mask=bottom_mask)
        else:
            preds = F.METHODS[name](origin.Y_fit, cfg.horizon, cfg.seasonal_period)
        origin.forecasts[name] = np.maximum(preds, 0.0)
        origin.timings[name] = time.perf_counter() - t0


def residual_variance(validation: Origin, method: str) -> np.ndarray:
    """Out of sample error variance per node, from the validation origin.

    Using held out errors rather than in sample ones keeps the variance estimate
    honest for methods like the pooled gradient boosting model, whose in sample
    residuals are optimistic in a way that would quietly mis-weight the whole
    reconciliation.
    """
    err = validation.Y_actual - validation.forecasts[method]
    return err.var(axis=1, ddof=0)


def reconcile_all(h: Hierarchy, meta: pd.DataFrame, yhat: np.ndarray,
                  Y_fit: np.ndarray, resid_var: np.ndarray) -> dict[str, np.ndarray]:
    """`Y_fit` is the full node history; the proportions need its bottom rows."""
    B_fit = Y_fit[h.rows_for("bottom")]
    props = R.historical_proportions(B_fit)
    within = R.proportions_within(B_fit, meta, h.bottom_ids, MIDDLE_LEVEL)
    return {
        "base": yhat,
        "bottom_up": R.bottom_up(h, yhat),
        "top_down": R.top_down(h, yhat, props),
        "middle_out": R.middle_out(h, yhat, meta, MIDDLE_LEVEL, within),
        "ols": R.projection(h, yhat, "ols"),
        "wls_struct": R.projection(h, yhat, "wls_struct"),
        "wls_var": R.projection(h, yhat, "wls_var", residual_var=resid_var),
    }


def score(h: Hierarchy, actual: np.ndarray, pred: np.ndarray, Y_fit: np.ndarray,
          season: int) -> pd.DataFrame:
    """RMSSE and MASE per node, tagged with its level."""
    scale = M.seasonal_naive_scale(Y_fit, season)
    return pd.DataFrame({
        "node_id": h.nodes["node_id"].to_numpy(),
        "level": h.nodes["level"].to_numpy(),
        "rmsse": M.rmsse(actual, pred, Y_fit, season),
        "mase": M.mase(actual, pred, scale),
    })


def by_level(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = scored.groupby("level", observed=True)
    out = grouped.agg(nodes=("rmsse", "size"),
                      rmsse=("rmsse", "mean"),
                      mase=("mase", "mean"),
                      rmsse_sd=("rmsse", "std")).reset_index()
    out["rmsse_se"] = out["rmsse_sd"] / np.sqrt(out["nodes"])
    return out.drop(columns="rmsse_sd")
