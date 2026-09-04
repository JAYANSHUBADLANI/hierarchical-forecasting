"""Config loading, and the paths every module resolves against.

Paths in the config file are relative to the repository root so nothing in a
committed file depends on where the repository happens to sit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def repo_path(relative: str | Path) -> Path:
    return ROOT / relative


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def levels(self) -> list[str]:
        return list(self.raw["hierarchy"]["levels"])

    @property
    def expected_counts(self) -> dict[str, int]:
        return dict(self.raw["hierarchy"]["expected_counts"])

    @property
    def horizon(self) -> int:
        return int(self.raw["evaluation"]["horizon"])

    @property
    def validation_horizon(self) -> int:
        return int(self.raw["evaluation"]["validation_horizon"])

    @property
    def seasonal_period(self) -> int:
        return int(self.raw["evaluation"]["seasonal_period"])

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(repo_path(path)) as handle:
        return Config(yaml.safe_load(handle))
