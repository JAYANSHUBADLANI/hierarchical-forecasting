"""Figures, all drawn from the committed tables so they cannot drift from them."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hierforecast.config import repo_path  # noqa: E402

TABLES, FIGS = repo_path("outputs/tables"), repo_path("outputs/figures")
LEVEL_ORDER = ["total", "state", "store", "store_cat", "store_dept", "bottom"]
PATTERNS = ["smooth", "erratic", "intermittent", "lumpy"]


def accuracy_by_level() -> None:
    d = pd.read_csv(TABLES / "accuracy_by_level.csv")
    bases = sorted(d.base.unique())
    fig, axes = plt.subplots(1, len(bases), figsize=(15, 4), sharey=True)
    x = np.arange(len(LEVEL_ORDER))
    for ax, base in zip(axes, bases):
        sub = d[d.base == base]
        for recon in ["base", "bottom_up", "wls_struct", "wls_var", "ols", "top_down"]:
            r = sub[sub.reconciliation == recon].set_index("level").reindex(LEVEL_ORDER)
            ax.plot(x, r["rmsse"], marker="o", ms=4, lw=1.4, label=recon,
                    ls="--" if recon == "base" else "-")
        ax.set_title(base); ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_ORDER, rotation=45, ha="right", fontsize=8)
        ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("RMSSE (lower is better)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Accuracy at every level, by base method and reconciliation", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "accuracy_by_level.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)


def incoherence() -> None:
    d = pd.read_csv(TABLES / "incoherence.csv").sort_values("gap_pct_of_root")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colours = ["0.6" if c else "#b4442e" for c in d.already_coherent]
    ax.barh(d.method, d.gap_pct_of_root, color=colours)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("root forecast minus sum of its own bottom forecasts, percent of root")
    ax.set_title("How far the base forecasts are from adding up")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "incoherence.png", dpi=140); plt.close(fig)


def pattern_slice() -> None:
    d = pd.read_csv(TABLES / "accuracy_by_pattern.csv")
    base_rows = d[d.reconciliation == "base"].set_index(["base", "pattern"])["rmsse"]
    best = (d[d.reconciliation != "base"]
            .loc[d[d.reconciliation != "base"].groupby(["base", "pattern"])["rmsse"].idxmin()]
            .set_index(["base", "pattern"]))
    gain = (100 * (base_rows - best["rmsse"]) / base_rows).unstack()[PATTERNS]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(PATTERNS)); w = 0.2
    for i, base in enumerate(gain.index):
        ax.bar(x + (i - 1.5) * w, gain.loc[base], w, label=base)
    ax.set_xticks(x); ax.set_xticklabels(PATTERNS)
    ax.set_ylabel("best reconciliation gain over base, percent")
    ax.axhline(0, color="0.3", lw=1); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Does reconciliation help less where demand is intermittent")
    fig.tight_layout(); fig.savefig(FIGS / "gain_by_pattern.png", dpi=140); plt.close(fig)
    gain.round(2).to_csv(TABLES / "gain_by_pattern.csv")


def intermittency_classes() -> None:
    prof = pd.read_parquet(TABLES / "intermittency_profile.parquet")
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sample = prof.sample(min(6000, len(prof)), random_state=0)
    ax.scatter(sample.adi.clip(upper=20), sample.cv2.clip(upper=6), s=3, alpha=0.25,
               c="#2f6f8f", linewidths=0)
    ax.axvline(1.32, color="#b4442e", lw=1.2); ax.axhline(0.49, color="#b4442e", lw=1.2)
    ax.set_xlabel("average demand interval (clipped at 20)")
    ax.set_ylabel("squared coefficient of variation (clipped at 6)")
    counts = prof.pattern.value_counts()
    ax.set_title("Demand patterns at the bottom level\n"
                 + ", ".join(f"{k} {v:,}" for k, v in counts.items()), fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "demand_patterns.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    accuracy_by_level(); incoherence(); pattern_slice(); intermittency_classes()
    print("figures written to outputs/figures")
