"""Render network-structure and time-horizon sensitivity supplementary figures.

Supplementary Figure S6 (network structure): median mortality reduction
(95% UI across parameter draws) for the main strategies, estimated on three
contact networks spanning low to realistic clustering (two-layer, clustering
~0.03; intermediate, ~0.40; primary three-layer clustered, ~0.50).

Supplementary Figure S7 (time horizon): outcomes at 90 and 180 days
from 500 paired replicates per strategy. Panel A shows median deaths;
panel B shows the median paired mortality reduction vs the
no-vaccination base-operations comparator, matched on replicate id.

Outputs (PNG 300 dpi + vector PDF):
  figures/final/Supplementary_Figure_S6.png/.pdf
  figures/final/Supplementary_Figure_S7.png/.pdf
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/final"

NETWORKS = [
    ("figure2_psa_acceptance_old_20260804", "Two-layer (clustering ≈ 0.03)", "#8A94A6", "o"),
    ("figure2_psa_acceptance_mid_20260804", "Intermediate (clustering ≈ 0.40)", "#D8A23C", "s"),
    ("figure2_psa_acceptance_G1_20260804", "Primary clustered (clustering ≈ 0.50)", "#21918C", "^"),
]

S6_STRATEGIES = [
    ("vax_base_ops", "Ring 2 Vax Alone (Base Ops)"),
    ("comm_base_20", "Community Vax 20% (Base Ops)"),
    ("comm_base_40", "Community Vax 40% (Base Ops)"),
    ("comm_base_60", "Community Vax 60% (Base Ops)"),
    ("comm_base_80", "Community Vax 80% (Base Ops)"),
    ("no_vax_enh_ops", "Enhanced Ops Alone"),
    ("vax_enh_ops", "Enhanced Ops + Ring 2 Vax"),
    ("incremental_ring_vax", "Incremental Ring 2 Vax (vs Enh)"),
]

HORIZON_CSV = ROOT / "data_and_results/outputs/horizon_extension_paired_20260804/horizon_extension_raw.csv"
HORIZONS = [90, 180]
S7_STRATEGIES = [
    ("no_vax_base_ops", "No vaccination (base ops)", "#172033"),
    ("no_vax_enh_ops", "No vaccination (enhanced ops)", "#667085"),
    ("vax_base_ops", "Ring 2 vax (base ops)", "#2878B5"),
    ("vax_enh_ops", "Ring 2 vax (enhanced ops)", "#5DADE2"),
    ("comm_base_40", "Community vax 40%", "#2A9288"),
    ("comm_base_60", "Community vax 60%", "#8FD0C6"),
]


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#E1E4E8", lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')} / {stem.with_suffix('.pdf')}")


def render_s6():
    summaries = {}
    for dirname, label, color, marker in NETWORKS:
        path = ROOT / "data_and_results/outputs" / dirname / "figure2_values.csv"
        with path.open(newline="") as handle:
            summaries[label] = {row["scenario"]: row for row in csv.DictReader(handle)}

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    y_base = np.arange(len(S6_STRATEGIES))[::-1]
    offsets = np.linspace(-0.26, 0.26, len(NETWORKS))
    for (dirname, label, color, marker), offset in zip(NETWORKS, offsets):
        data = summaries[label]
        for y, (key, _) in zip(y_base, S6_STRATEGIES):
            row = data[key]
            median = float(row["median_deaths_averted_pct"])
            low = float(row["ui_low_95"])
            high = float(row["ui_high_95"])
            ax.plot([low, high], [y + offset, y + offset], color=color, lw=1.6, alpha=0.85, zorder=2)
            ax.scatter(median, y + offset, s=64, color=color, marker=marker, zorder=3,
                       edgecolor="white", linewidth=0.6)
    # Legend built from proxy artists
    for label, color, marker in [(l, c, m) for _, l, c, m in NETWORKS]:
        ax.scatter([], [], s=64, color=color, marker=marker, label=label)
    ax.legend(frameon=False, fontsize=10, loc="lower right", title="Contact network", title_fontsize=10)
    ax.set_yticks(y_base, [label for _, label in S6_STRATEGIES], fontsize=10.5, fontweight="bold")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-0.6, len(S6_STRATEGIES) - 0.4)
    ax.set_xlabel("Median mortality reduction vs comparator (%, 95% UI)", fontweight="bold")
    style(ax, grid_axis="x")
    save(fig, OUT / "Supplementary_Figure_S6")


def render_s7():
    df = pd.read_csv(HORIZON_CSV)
    strategies = [key for key, _, _ in S7_STRATEGIES]

    # Paired per-replicate mortality reduction vs no_vax_base_ops, matched on
    # replicate id within each horizon.
    base = df[df.strategy == "no_vax_base_ops"].set_index(["horizon", "replicate_id"])["deaths"]
    paired = []
    for key, label, _ in S7_STRATEGIES:
        if key == "no_vax_base_ops":
            continue
        sub = df[df.strategy == key].set_index(["horizon", "replicate_id"])
        common = sub.index.intersection(base.index)
        reduction = 100 * (base.loc[common].values - sub.loc[common, "deaths"].values) / base.loc[common].values
        tmp = pd.DataFrame({"horizon": [h for h, _ in common], "reduction": reduction})
        tmp["strategy"] = key
        paired.append(tmp)
    paired = pd.concat(paired)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"wspace": 0.28})

    # Panel A: median deaths by strategy and horizon (log scale)
    ax = axes[0]
    width = 0.13
    x = np.arange(len(HORIZONS))
    for i, (key, label, color) in enumerate(S7_STRATEGIES):
        meds = [df[(df.strategy == key) & (df.horizon == h)]["deaths"].median() for h in HORIZONS]
        ax.bar(x + (i - 2.5) * width, meds, width=width * 0.92, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{h} days" for h in HORIZONS])
    ax.set_ylabel("Median total deaths (log scale)", fontweight="bold")
    ax.set_xlabel("Simulation horizon", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    style(ax)
    ax.text(-0.10, 1.03, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Panel B: median paired mortality reduction vs no-vax base ops
    ax = axes[1]
    paired_strategies = [s for s in S7_STRATEGIES if s[0] != "no_vax_base_ops"]
    for i, (key, label, color) in enumerate(paired_strategies):
        meds, los, his = [], [], []
        for h in HORIZONS:
            vals = paired[(paired.strategy == key) & (paired.horizon == h)]["reduction"]
            meds.append(vals.median())
            los.append(np.percentile(vals, 2.5))
            his.append(np.percentile(vals, 97.5))
        xi = x + (i - 2) * (width * 1.15)
        ax.errorbar(xi, meds, yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                    fmt="o", color=color, ms=6.5, lw=1.6, capsize=3,
                    ecolor=color, elinewidth=1.1, label=label)
        ax.plot(xi, meds, color=color, lw=1.2, alpha=0.55)
    ax.axhline(0, color="#98A2B3", lw=1, ls="--")
    ax.set_xticks(x, [f"{h} days" for h in HORIZONS])
    ax.set_ylabel("Paired mortality reduction vs no vaccination\n(base ops; median, 95% range) (%)", fontweight="bold")
    ax.set_xlabel("Simulation horizon", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    style(ax)
    ax.text(-0.10, 1.03, "B", transform=ax.transAxes, fontsize=14, fontweight="bold")

    save(fig, OUT / "Supplementary_Figure_S7")

    # Console summary for spot-checking
    for h in HORIZONS:
        med = df[(df.strategy == "no_vax_base_ops") & (df.horizon == h)]["deaths"].median()
        print(f"horizon {h}: no_vax_base_ops median deaths = {med:.0f}")
    for key, label, _ in paired_strategies:
        vals = paired[(paired.strategy == key) & (paired.horizon == 180)]["reduction"]
        print(f"horizon 180 (check): {key} median paired reduction = {vals.median():.1f}%")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)
    render_s6()
    render_s7()


if __name__ == "__main__":
    main()
