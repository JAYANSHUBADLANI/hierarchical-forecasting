"""Scale free error measures.

Percentage errors are unusable on this data. Nearly sixty percent of the bottom
level observations are zero, so MAPE is either undefined or explodes, and any
table built on it would be dominated by whichever series happened to have the
smallest non zero denominator. So everything here is scaled against an in sample
seasonal naive error, which is the standard way to compare across series of very
different volume.

The denominator is computed on the fitting window only, never on the window being
scored, and for a bottom level series only over the days it was actually live.
"""
from __future__ import annotations

import numpy as np


def seasonal_naive_scale(train: np.ndarray, season: int,
                         live: np.ndarray | None = None) -> np.ndarray:
    """Mean absolute seasonal difference per series, the MASE denominator.

    Returns NaN where a series has too little history to form one seasonal
    difference, so those series can be excluded from the average rather than
    silently contributing an infinite error.
    """
    if train.shape[1] <= season:
        return np.full(train.shape[0], np.nan)
    diff = np.abs(train[:, season:] - train[:, :-season])
    if live is None:
        return diff.mean(axis=1)

    valid = live[:, season:] & live[:, :-season]
    counts = valid.sum(axis=1)
    totals = np.where(valid, diff, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return scale


def mase(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Mean absolute scaled error per series."""
    err = np.abs(actual - pred).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, err / scale, np.nan)


def rmsse(actual: np.ndarray, pred: np.ndarray, train: np.ndarray, season: int,
          live: np.ndarray | None = None) -> np.ndarray:
    """Root mean squared scaled error, the squared error analogue of MASE."""
    if train.shape[1] <= season:
        return np.full(train.shape[0], np.nan)
    sq = (train[:, season:] - train[:, :-season]) ** 2
    if live is None:
        denom = sq.mean(axis=1)
    else:
        valid = live[:, season:] & live[:, :-season]
        counts = valid.sum(axis=1)
        denom = np.where(counts > 0, np.where(valid, sq, 0.0).sum(axis=1)
                         / np.maximum(counts, 1), np.nan)
    num = ((actual - pred) ** 2).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, np.sqrt(num / denom), np.nan)
