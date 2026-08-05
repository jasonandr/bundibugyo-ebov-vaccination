"""Render a review-only Figure 2 forest plot from the current PSA summary."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    ("vax_base_ops", "Ring 2 Vax Alone (Base Ops)", "#2878B5"),
    ("comm_base_20", "Community Vax 20% (Base Ops)", "#21918c"),
    ("comm_base_40", "Community Vax 40% (Base Ops)", "#21918c"),
    ("comm_base_60", "Community Vax 60% (Base Ops)", "#21918c"),
    ("comm_base_80", "Community Vax 80% (Base Ops)", "#21918c"),
    ("no_vax_enh_ops", "Enhanced Ops Alone", "#56616f"),
    ("vax_enh_ops", "Enhanced Ops + Ring 2 Vax", "#2878B5"),
    ("incremental_ring_vax", "Incremental Ring 2 Vax (vs Enh)", "#CA6D4B"),
]


def centred_decimal(value):
    return f"{value:.1f}".replace(".", "·")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    with args.summary.open(newline="") as handle:
        data = {row["scenario"]: row for row in csv.DictReader(handle)}
    missing = [key for key, _, _ in ROWS if key not in data]
    if missing:
        raise ValueError(f"Summary is missing scenarios: {missing}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    positions = np.arange(len(ROWS))[::-1]
    for y, (key, label, color) in zip(positions, ROWS):
        row = data[key]
        median = float(row["median_deaths_averted_pct"])
        low = float(row["ui_low_95"])
        high = float(row["ui_high_95"])
        ax.hlines(y, low, high, color=color, lw=3.2)
        ax.scatter(median, y, s=175, color=color, zorder=3)
        text = f"{centred_decimal(median)}% ({centred_decimal(low)}–{centred_decimal(high)}%)"
        ax.text(104.7, y, text, va="center", ha="left", fontsize=12.5, fontweight="bold", color="#172033")

    ax.set_yticks(positions, [label for _, label, _ in ROWS], fontsize=12.5, fontweight="bold")
    ax.set_xlim(-10, 145)
    ax.set_ylim(-0.35, len(ROWS) - 0.65)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Mortality reduction vs comparator (%)", fontsize=13.5, fontweight="bold")
    ax.text(104.7, len(ROWS) - 0.35, "Median (95% UI)", fontsize=13.5, fontweight="bold", color="#172033")
    ax.grid(axis="x", color="#e1e4e8", lw=1.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.4)
    ax.tick_params(axis="both", width=1.4, length=6)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
