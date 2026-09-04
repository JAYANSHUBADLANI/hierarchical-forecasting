"""The summing matrix, and the check that the tree actually adds up.

Everything in reconciliation is written in terms of one matrix. If the bottom
level series are stacked into a vector b, then the full vector of all series at
all levels is S @ b, where S has one row per node and one column per bottom
series, and a one wherever that bottom series feeds that node.

I build S once, sparsely, and reuse it. Before anything is forecast I check the
tree adds up, by computing every aggregate twice through two independent paths,
a sparse matrix product and a pandas groupby, and requiring them to agree
exactly. Two paths agreeing is a real check. Building aggregates one way and
then asserting they sum is not, it would only be restating the construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Hierarchy:
    levels: list[str]
    nodes: pd.DataFrame          # node_id, level, row position in S
    S: sparse.csr_matrix         # (n_nodes, n_bottom)
    bottom_ids: list[str]

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_ids)

    def rows_for(self, level: str) -> np.ndarray:
        return self.nodes.index[self.nodes["level"] == level].to_numpy()

    def node_names(self, level: str) -> list[str]:
        return self.nodes.loc[self.nodes["level"] == level, "node_id"].tolist()


def build(meta: pd.DataFrame, levels: list[str]) -> Hierarchy:
    bottom_ids = meta["bottom"].tolist()
    col_of = {b: i for i, b in enumerate(bottom_ids)}

    node_rows, rows, cols = [], [], []
    for level in levels:
        for node, group in meta.groupby(level, sort=True):
            r = len(node_rows)
            node_rows.append((node, level))
            for b in group["bottom"]:
                rows.append(r)
                cols.append(col_of[b])

    n_nodes, n_bottom = len(node_rows), len(bottom_ids)
    S = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n_nodes, n_bottom),
    )
    nodes = pd.DataFrame(node_rows, columns=["node_id", "level"])
    return Hierarchy(levels=levels, nodes=nodes, S=S, bottom_ids=bottom_ids)


def bottom_matrix(panel: pd.DataFrame,
                  bottom_ids: list[str]) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Bottom level series as a dense (n_bottom, n_time) array, plus a live mask.

    The file I use has had the rows before each item's first recorded sale
    stripped out, so series start on 1,652 different dates and only 23 percent of
    them span the full history. I checked: not one series has a zero in its first
    row, which is what tells you the leading rows were removed rather than the
    item genuinely selling nothing.

    For aggregation those missing days must be zero, because an item that was not
    on sale contributed nothing to any total, and the sums have to stay exact.
    But zero because an item did not exist is not the same fact as zero because
    nobody bought it, and treating them alike would make a recently introduced
    item look far more intermittent than it is and would wreck any error measure
    scaled over the full history. So this returns the padded matrix for
    aggregation and a boolean mask of the days each series was actually live, and
    every per series calculation downstream uses the mask.
    """
    wide = panel.pivot(index="unique_id", columns="ds", values="y").reindex(bottom_ids)
    live = wide.notna().to_numpy()
    B = wide.fillna(0.0).to_numpy(dtype=np.float64)
    return B, pd.DatetimeIndex(wide.columns), live


def first_live_index(live: np.ndarray) -> np.ndarray:
    """Column index of each series' first live day."""
    return live.argmax(axis=1)


def aggregate(h: Hierarchy, B: np.ndarray) -> np.ndarray:
    """All levels at once: (n_nodes, n_time)."""
    return h.S @ B


def check_coherence(h: Hierarchy, B: np.ndarray, meta: pd.DataFrame,
                    panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute every level by groupby and require exact agreement with S @ B.

    Also checks each level totals to the same grand total, which catches a node
    that is missing from a level or counted twice.
    """
    Y = aggregate(h, B)
    cols = list(dict.fromkeys(["bottom", *h.levels]))
    keyed = panel.merge(meta[cols], left_on="unique_id", right_on="bottom", how="left")
    records = []
    for level in h.levels:
        by_groupby = (keyed.groupby([level, "ds"], observed=True)["y"].sum()
                      .unstack(fill_value=0.0).sort_index())
        rows = h.rows_for(level)
        by_matrix = pd.DataFrame(Y[rows], index=h.node_names(level),
                                 columns=None).sort_index()
        by_matrix.columns = by_groupby.columns
        max_abs_diff = float(np.abs(by_matrix.to_numpy() - by_groupby.to_numpy()).max())
        records.append({
            "level": level,
            "nodes": len(rows),
            "grand_total": float(Y[rows].sum()),
            "max_abs_diff_vs_groupby": max_abs_diff,
            "exact": max_abs_diff == 0.0,
        })
    out = pd.DataFrame(records)
    totals = out["grand_total"].round(6).nunique()
    out["all_levels_same_total"] = totals == 1
    return out
