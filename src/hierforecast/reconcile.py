"""Reconciliation, written so it works at 30,490 bottom series.

Every method here is the same shape. Base forecasts for all nodes, yhat, are
mapped to a bottom level vector by some matrix G, and then pushed back up:

    ytilde = S @ (G @ yhat)

The methods differ only in G. Bottom up ignores every aggregate. Top down uses
only the root and splits it by historical proportions. Middle out does the same
from a chosen middle level, aggregating below it and splitting above. The
projection methods pick the bottom vector that is closest to the base forecasts
in a weighted least squares sense, which gives the normal equations

    (S' W^-1 S) b = S' W^-1 yhat

**Why this is not solved by forming a matrix.** The obvious route is to build
S' W^-1 S and invert it. Here that matrix is 30,490 by 30,490, about 7.4 GB
dense, and it is dense rather than sparse because every pair of bottom series
shares the root node. So instead the normal equations are solved by conjugate
gradient against an operator that never forms the matrix: one application costs
two sparse products with S, which is 183,000 non zeros. The system is symmetric
positive definite, so CG is the right solver, and the result is the exact
projection to the tolerance requested rather than an approximation of it.

The weighting W is where the estimators differ:

- ``ols``      W = I. Treats a store forecast error and a single item forecast
               error as equally important, which is clearly wrong but is the
               reference point everything else is measured against.
- ``wls_struct`` W = diag of the row sums of S, so a node is weighted by how many
               bottom series it aggregates. Needs no residuals at all.
- ``wls_var``  W = diag of in sample one step residual variances. This is MinT
               with a diagonal covariance, and it is as far as a covariance can
               be taken here without the shrinkage machinery below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .hierarchy import Hierarchy


def _solve_normal_equations(S: sparse.csr_matrix, w_inv: np.ndarray,
                            yhat: np.ndarray, tol: float = 1e-10,
                            maxiter: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares bottom vector, one column of the horizon at a time.

    Jacobi preconditioned, and it needs to be. With variance weights the node
    weights span many orders of magnitude, a single item in one store against the
    grand total, and plain CG does not converge in any sensible number of steps.
    The preconditioner is the diagonal of the normal matrix, which is free to
    compute exactly here: every entry of S is one, so the diagonal is just the sum
    of the inverse weights of the nodes each bottom series feeds, S' w_inv.

    Returns the solution and the iteration count per horizon step, which is
    reported rather than discarded because it is the honest diagnostic for whether
    a weighting has made the system hard.
    """
    St = S.T.tocsr()
    n_bottom = S.shape[1]

    def matvec(x: np.ndarray) -> np.ndarray:
        return St @ (w_inv * (S @ x))

    diag = np.asarray(St @ w_inv).ravel()
    diag = np.where(diag > 0, diag, 1.0)
    precond = LinearOperator((n_bottom, n_bottom), matvec=lambda x: x / diag,
                             dtype=np.float64)
    op = LinearOperator((n_bottom, n_bottom), matvec=matvec, dtype=np.float64)

    out = np.zeros((n_bottom, yhat.shape[1]))
    iters = np.zeros(yhat.shape[1], dtype=int)
    for j in range(yhat.shape[1]):
        count = 0

        def _count(_):
            nonlocal count
            count += 1

        rhs = St @ (w_inv * yhat[:, j])
        sol, info = cg(op, rhs, rtol=tol, maxiter=maxiter, M=precond, callback=_count)
        if info != 0:
            raise RuntimeError(
                f"CG did not converge for horizon step {j}, info={info}. "
                "The weighting has made the normal equations too ill conditioned.")
        out[:, j] = sol
        iters[j] = count
    return out, iters


def bottom_up(h: Hierarchy, yhat: np.ndarray) -> np.ndarray:
    rows = h.rows_for("bottom")
    return h.S @ yhat[rows]


def top_down(h: Hierarchy, yhat: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    """Split the root forecast by fixed historical proportions.

    The proportions are the share of the grand total each bottom series held over
    the fitting window. That is one of several standard disaggregation rules and
    the choice is doing real work in the result, so it is named here and stated
    in the README rather than buried.
    """
    root = h.rows_for("total")[0]
    b = np.outer(proportions, yhat[root])
    return h.S @ b


def middle_out(h: Hierarchy, yhat: np.ndarray, meta: pd.DataFrame,
               level: str, proportions_within: np.ndarray) -> np.ndarray:
    """Trust one middle level, aggregate above it, split below it."""
    node_names = h.node_names(level)
    pos = {n: i for i, n in enumerate(node_names)}
    rows = h.rows_for(level)
    parent_of_bottom = meta.set_index("bottom").loc[h.bottom_ids, level].to_numpy()
    parent_idx = np.array([pos[p] for p in parent_of_bottom])
    b = yhat[rows][parent_idx] * proportions_within[:, None]
    return h.S @ b


def projection(h: Hierarchy, yhat: np.ndarray, weights: str = "ols",
               residual_var: np.ndarray | None = None) -> np.ndarray:
    if weights == "ols":
        w = np.ones(h.S.shape[0])
    elif weights == "wls_struct":
        w = np.asarray(h.S.sum(axis=1)).ravel()
    elif weights == "wls_var":
        if residual_var is None:
            raise ValueError("wls_var needs in sample residual variances")
        w = np.maximum(residual_var, 1e-8)
    else:
        raise ValueError(f"unknown weighting {weights!r}")

    b, iters = _solve_normal_equations(h.S, 1.0 / w, yhat)
    projection.last_cg_iterations = iters
    return h.S @ b


def historical_proportions(B_fit: np.ndarray) -> np.ndarray:
    """Each bottom series' share of the grand total over the fitting window."""
    totals = B_fit.sum(axis=1)
    grand = totals.sum()
    if grand <= 0:
        return np.full(B_fit.shape[0], 1.0 / B_fit.shape[0])
    return totals / grand


def proportions_within(B_fit: np.ndarray, meta: pd.DataFrame, bottom_ids: list[str],
                       level: str) -> np.ndarray:
    """Each bottom series' share of its own parent at the given level."""
    totals = pd.Series(B_fit.sum(axis=1), index=bottom_ids)
    parent = meta.set_index("bottom").loc[bottom_ids, level]
    parent_total = totals.groupby(parent).transform("sum")
    share = np.where(parent_total > 0, totals / parent_total.replace(0, np.nan), 0.0)
    return np.nan_to_num(share)


def is_coherent(h: Hierarchy, ytilde: np.ndarray, tol: float = 1e-6) -> tuple[bool, float]:
    """A reconciliation that does not reconcile is a bug that is easy to miss."""
    rows = h.rows_for("bottom")
    implied = h.S @ ytilde[rows]
    err = float(np.abs(implied - ytilde).max())
    return err <= tol, err
