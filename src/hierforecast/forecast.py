"""Base forecasts, produced independently at every level of the hierarchy.

Everything here is vectorised across series rather than looped over them. There
are 30,604 nodes and several methods and two forecast origins, so a per series
Python loop would turn this project into an overnight job for no gain. Where a
method is a recursion in time, the loop runs over time steps and updates every
series at once.

I implement these rather than importing a forecasting package because the point
of the project is the comparison, and a baseline I cannot inspect is not a
baseline I can defend.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def seasonal_naive(train: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Repeat the last full season forward."""
    if train.shape[1] < season:
        return np.repeat(train[:, -1:], horizon, axis=1)
    last = train[:, -season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last, (1, reps))[:, :horizon]


def _ses_level(train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Final simple exponential smoothing level, vectorised over series."""
    level = train[:, 0].copy()
    for t in range(1, train.shape[1]):
        level = alpha * train[:, t] + (1.0 - alpha) * level
    return level


def ses(train: np.ndarray, horizon: int,
        alpha_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simple exponential smoothing with alpha chosen per series.

    Alpha is picked from a grid by in sample sum of squared one step errors,
    which is the usual criterion and keeps the whole thing vectorised. Returns
    the flat forecast and the chosen alphas, because the alpha distribution is
    worth reporting: an alpha pinned at the bottom of the grid means the method
    has decided the series is noise around a constant.
    """
    if alpha_grid is None:
        alpha_grid = np.arange(0.05, 1.0, 0.05)

    n = train.shape[0]
    best_sse = np.full(n, np.inf)
    best_alpha = np.full(n, alpha_grid[0])

    for a in alpha_grid:
        level = train[:, 0].copy()
        sse = np.zeros(n)
        for t in range(1, train.shape[1]):
            err = train[:, t] - level
            sse += err * err
            level = a * train[:, t] + (1.0 - a) * level
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        best_alpha = np.where(better, a, best_alpha)

    level = _ses_level(train, best_alpha)
    return np.repeat(level[:, None], horizon, axis=1), best_alpha


def croston(train: np.ndarray, horizon: int, alpha: float = 0.1,
            variant: str = "sba") -> np.ndarray:
    """Croston's method for intermittent demand, optionally SBA corrected.

    Croston smooths the size of non zero demands and the interval between them
    separately, and forecasts their ratio. The plain ratio is known to be biased
    upward; the Syntetos and Boylan approximation multiplies it by (1 - alpha/2).
    Both are available because showing the bias correction matter is part of the
    point.

    The recursion runs over time with every series updated together.
    """
    n, T = train.shape
    nz = train > 0

    has_any = nz.any(axis=1)
    first = np.where(has_any, nz.argmax(axis=1), 0)

    # Initialise the interval level from the first interval actually observed,
    # not from 1. Starting at 1 when the true interval is, say, 4 biases the rate
    # upward for as long as the smoother takes to climb, which at alpha 0.1 is
    # most of a short series. Series with a single sale get their observed span,
    # which is the only interval evidence they have.
    second = np.full(n, -1)
    for i in range(n):
        if not has_any[i]:
            continue
        rest = np.flatnonzero(nz[i, first[i] + 1:])
        if rest.size:
            second[i] = first[i] + 1 + rest[0]

    z = np.where(has_any, train[np.arange(n), first], 0.0)   # demand size level
    p = np.where(second >= 0, second - first,
                 np.maximum(T - first, 1.0)).astype(np.float64)
    p = np.where(has_any, p, 1.0)
    since = np.ones(n)
    started = np.zeros(n, dtype=bool)

    for t in range(T):
        is_first = has_any & (t == first)
        started |= is_first
        sale = nz[:, t] & started & ~is_first
        if sale.any():
            z = np.where(sale, alpha * train[:, t] + (1 - alpha) * z, z)
            p = np.where(sale, alpha * since + (1 - alpha) * p, p)
        since = np.where(sale | is_first, 1.0, since + 1.0)

    rate = np.where(has_any, z / np.maximum(p, EPS), 0.0)
    if variant == "sba":
        rate = rate * (1.0 - alpha / 2.0)
    elif variant != "croston":
        raise ValueError(f"unknown croston variant {variant!r}")
    return np.repeat(rate[:, None], horizon, axis=1)


METHODS = {
    "seasonal_naive": lambda tr, h, s: seasonal_naive(tr, h, s),
    "ses": lambda tr, h, s: ses(tr, h)[0],
    "croston_sba": lambda tr, h, s: croston(tr, h, variant="sba"),
}
