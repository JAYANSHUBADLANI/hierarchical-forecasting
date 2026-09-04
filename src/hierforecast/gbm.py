"""A pooled gradient boosting arm.

One model is fitted across all nodes rather than one per node. Fitting 30,604
separate models would be slower and worse: most nodes are sparse, and a pooled
model borrows the day of week and lag structure that they share.

Two choices worth stating because they shape the result. Training uses a recent
window rather than the whole history, since a 2011 pattern is weak evidence about
2016 and the full panel is 59 million rows. And the bottom level is row sampled,
because it outnumbers every aggregate level by two orders of magnitude and would
otherwise be the only thing the model learns. Both are configurable and both are
reported.

Forecasts are produced recursively over the horizon, feeding each day's
prediction back in as the lag for the next, which is what makes the horizon
honest: at day 28 the model has not been shown any actual value it would not
have had.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LAGS = (7, 14, 28)
WINDOWS = (7, 28)


def _features(hist: np.ndarray, dates: pd.DatetimeIndex, t: int,
              level_code: np.ndarray) -> np.ndarray:
    """Feature block for every node at time index t, using history before t."""
    cols = [np.full(hist.shape[0], dates[t].dayofweek, dtype=np.float64),
            np.full(hist.shape[0], dates[t].day, dtype=np.float64),
            np.full(hist.shape[0], dates[t].month, dtype=np.float64),
            np.full(hist.shape[0], dates[t].dayofyear, dtype=np.float64),
            level_code.astype(np.float64)]
    for lag in LAGS:
        cols.append(hist[:, t - lag] if t - lag >= 0 else np.zeros(hist.shape[0]))
    for w in WINDOWS:
        lo = max(0, t - w)
        cols.append(hist[:, lo:t].mean(axis=1) if t > lo else np.zeros(hist.shape[0]))
    return np.column_stack(cols)


FEATURE_NAMES = (["dow", "dom", "month", "doy", "level"]
                 + [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in WINDOWS])


def fit_predict(Y: np.ndarray, dates: pd.DatetimeIndex, horizon: int,
                level_code: np.ndarray, seed: int, train_days: int = 730,
                bottom_mask: np.ndarray | None = None,
                bottom_sample: int = 4000) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 909]))

    rows = (bottom_mask if bottom_mask is not None
            else np.zeros(Y.shape[0], dtype=bool))
    keep = ~rows
    if rows.any():
        idx = np.flatnonzero(rows)
        chosen = rng.choice(idx, size=min(bottom_sample, idx.size), replace=False)
        keep = keep.copy()
        keep[chosen] = True

    start = max(max(LAGS), Y.shape[1] - train_days)
    X_parts, y_parts = [], []
    for t in range(start, Y.shape[1]):
        X_parts.append(_features(Y, dates, t, level_code)[keep])
        y_parts.append(Y[keep, t])
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_depth=6,
        early_stopping=False, random_state=seed)
    model.fit(X, y)

    hist = Y.copy()
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = np.zeros((Y.shape[0], horizon))
    for i, d in enumerate(future):
        hist = np.column_stack([hist, np.zeros(hist.shape[0])])
        all_dates = dates.append(pd.DatetimeIndex(future[:i + 1]))
        t = hist.shape[1] - 1
        p = np.maximum(model.predict(_features(hist, all_dates, t, level_code)), 0.0)
        preds[:, i] = p
        hist[:, t] = p

    info = {"train_rows": int(X.shape[0]), "nodes_used": int(keep.sum()),
            "train_days": int(Y.shape[1] - start), "features": FEATURE_NAMES}
    return preds, info
