"""Loading the M5 files and recovering the hierarchy from the series id.

The mirror I use ships long format parquet with three columns, `unique_id`, `ds`
and `y`. The hierarchy is not in columns, it is encoded in the id string, so the
first job is to parse it back out.

An id looks like `FOODS_1_001_CA_1` and splits on underscores into five parts:
category, department number, item number, state, store number. From those five
parts every level of the tree is recoverable.
"""
from __future__ import annotations

import pandas as pd

from .config import Config, repo_path

ID_PARTS = 5


def parse_ids(ids: pd.Series) -> pd.DataFrame:
    """Split series ids into the hierarchy keys.

    Raises if any id does not have exactly five underscore separated parts,
    because a silently mis-parsed id would put a series under the wrong parent
    and every number after that would be measuring the bug.
    """
    parts = ids.astype(str).str.split("_", expand=True)
    if parts.shape[1] != ID_PARTS:
        raise ValueError(f"expected {ID_PARTS} id parts, got {parts.shape[1]}")
    if parts.isna().any().any():
        bad = ids[parts.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"ids with missing parts, for example {bad}")

    cat = parts[0]
    dept = parts[0] + "_" + parts[1]
    item = dept + "_" + parts[2]
    state = parts[3]
    store = state + "_" + parts[4]

    return pd.DataFrame(
        {
            "unique_id": ids.astype(str).to_numpy(),
            "cat_id": cat.to_numpy(),
            "dept_id": dept.to_numpy(),
            "item_id": item.to_numpy(),
            "state_id": state.to_numpy(),
            "store_id": store.to_numpy(),
        }
    )


def level_keys(meta: pd.DataFrame) -> pd.DataFrame:
    """Add one column per hierarchy level holding that level's node name."""
    out = meta.copy()
    out["total"] = "TOTAL"
    out["state"] = out["state_id"]
    out["store"] = out["store_id"]
    out["store_cat"] = out["store_id"] + "|" + out["cat_id"]
    out["store_dept"] = out["store_id"] + "|" + out["dept_id"]
    out["bottom"] = out["unique_id"]
    return out


def load_panel(cfg: Config, which: str = "train") -> pd.DataFrame:
    key = "train_parquet" if which == "train" else "test_parquet"
    frame = pd.read_parquet(repo_path(cfg["data"][key]))
    frame["unique_id"] = frame["unique_id"].astype(str)
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype("float64")
    return frame.sort_values(["unique_id", "ds"], ignore_index=True)


def load_metadata_from_ids(ids: pd.Series) -> pd.DataFrame:
    """Hierarchy table for a set of series ids, in sorted id order."""
    unique = pd.Series(sorted(set(ids.astype(str))), name="unique_id")
    return level_keys(parse_ids(unique))


def load_metadata(cfg: Config, ids: pd.Series) -> pd.DataFrame:
    return load_metadata_from_ids(ids)
