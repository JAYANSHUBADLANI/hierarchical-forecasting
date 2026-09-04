"""Classifying demand patterns, on each series' own live window.

Syntetos and Boylan (2005) split series on two statistics: the average demand
interval, which is periods divided by number of non zero demands, and the
squared coefficient of variation of the non zero demand sizes. Their cutoffs are
1.32 and 0.49, giving smooth, intermittent, erratic and lumpy.

The subtlety in this dataset is which periods count. Series here start on 1,652
different dates because the rows before an item's first sale were stripped out,
and I pad those days with zero so the hierarchy still adds up. If the
classification then runs over the padded history, a recently introduced item
gets hundreds of invented zero days and its demand interval is inflated by a
preprocessing choice rather than by its demand. So this measures every series
over its live window only. The difference that makes is reported, because it is
large enough to change the story.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH, INTERMITTENT, ERRATIC, LUMPY = "smooth", "intermittent", "erratic", "lumpy"


def _stats_one(y: np.ndarray) -> tuple[float, float, int]:
    nz = y[y > 0]
    if nz.size == 0:
        return np.inf, np.nan, 0
    adi = y.size / nz.size
    mean = nz.mean()
    cv2 = float((nz.std(ddof=0) / mean) ** 2) if mean > 0 else np.nan
    return float(adi), cv2, int(nz.size)


def classify(adi: float, cv2: float, adi_t: float, cv2_t: float) -> str:
    if not np.isfinite(adi):
        return LUMPY
    hi_adi, hi_cv2 = adi >= adi_t, (cv2 >= cv2_t if np.isfinite(cv2) else False)
    if hi_adi and hi_cv2:
        return LUMPY
    if hi_adi:
        return INTERMITTENT
    if hi_cv2:
        return ERRATIC
    return SMOOTH


def profile(B: np.ndarray, live: np.ndarray, ids: list[str],
            adi_t: float, cv2_t: float, use_live_window: bool = True) -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(ids):
        y = B[i, live[i]] if use_live_window else B[i]
        adi, cv2, n_nz = _stats_one(y)
        rows.append({
            "unique_id": sid,
            "periods": int(y.size),
            "nonzero_periods": n_nz,
            "zero_share": float((y == 0).mean()) if y.size else np.nan,
            "adi": adi,
            "cv2": cv2,
            "mean_nonzero_demand": float(y[y > 0].mean()) if n_nz else 0.0,
            "pattern": classify(adi, cv2, adi_t, cv2_t),
        })
    return pd.DataFrame(rows)
