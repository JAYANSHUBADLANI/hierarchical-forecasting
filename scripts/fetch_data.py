"""Download the M5 mirror, and build the committed sample from it.

The raw files stay out of version control. The sample is one store, which is
enough for the test suite to run with no network call.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hierforecast.config import load_config, repo_path  # noqa: E402

DATASET = "marcozanotti/m5-competition-dataset-parquet"
FILES = ["m5_train.parquet", "m5_test.parquet"]


def fetch() -> None:
    raw = repo_path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (raw / name).exists():
            print(f"{name} already here, skipping")
            continue
        print(f"downloading {name}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-f", name,
             "--unzip", "-p", str(raw)],
            check=True,
        )


def build_sample() -> None:
    import pandas as pd

    cfg = load_config()
    store = cfg["data"]["sample_store"]
    out = repo_path("data/sample")
    out.mkdir(parents=True, exist_ok=True)
    for which, name in (("train", "m5_train.parquet"), ("test", "m5_test.parquet")):
        frame = pd.read_parquet(repo_path("data/raw") / name)
        frame["unique_id"] = frame["unique_id"].astype(str)
        keep = frame[frame.unique_id.str.endswith(f"_{store}")]
        if which == "train":
            # The sample only has to let the tests run without a download, so it
            # carries one store over the last two years rather than all of it.
            cutoff = pd.to_datetime(keep["ds"]).max() - pd.Timedelta(days=730)
            keep = keep[pd.to_datetime(keep["ds"]) > cutoff]
        target = out / f"sample_{which}_{store}.parquet"
        keep.to_parquet(target, index=False)
        print(f"{target.name}: {len(keep):,} rows, {keep.unique_id.nunique():,} series")


if __name__ == "__main__":
    fetch()
    build_sample()
