"""Render the two-panel community-vaccination implementation Figure 4."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data_and_results/outputs/paired_figure_grids_20260722"


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def median(rows, key):
    return float(np.median([float(row[key]) for row in rows]))


def main():
    paired = read(REVIEW / "raw/figure3ab_paired_20260722/fig3_community_ve_paired_raw.csv")
    base_rows = [r for r in paired if float(r["coverage"]) == 0 and abs(float(r["ve"]) - .45) < 1e-8]
    base_deaths = median(base_rows, "base_deaths")

    grid = read(REVIEW / "raw/community_coverage_delay_0_14_28_paired_20260722.csv")

    def reduction(rows):
        if "mortality_reduction_pct" in rows[0]:
            return median(rows, "mortality_reduction_pct")
        return 100 * (base_deaths - median(rows, "deaths")) / base_deaths

    delay_values = [0., 14., 28.]
    panel_a = []
    for delay in delay_values:
        panel_a.append(reduction([r for r in grid if float(r["delay"]) == delay and float(r["coverage"]) == .5]))

    coverage = np.array(sorted({float(r["coverage"]) for r in grid}))
    z = np.array([
        [reduction([r for r in grid if float(r["delay"]) == delay and float(r["coverage"]) == cov]) for cov in coverage]
        for delay in delay_values
    ])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"wspace": .35})

    colors = ["#2A9288", "#D1C273", "#CA6D4B"]
    labels = ["At declaration\n(day 0)", "Declaration\n+14 days", "Declaration\n+28 days"]
    bars = axes[0].barh(labels, panel_a, color=colors, height=.56)
    axes[0].invert_yaxis()
    axes[0].set(title="50% community vaccination coverage", xlabel="Median mortality reduction (%)", xlim=(0, 90))
    for bar, value, color in zip(bars, panel_a, colors):
        axes[0].text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.0f}%", va="center", color=color, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("implementation", ["#F6E7D0", "#D9D58E", "#77A77E", "#2A6978"])
    im = axes[1].imshow(z, aspect="auto", cmap=cmap, origin="upper", vmin=20, vmax=90)
    axes[1].set(
        title="Coverage and implementation delay",
        xlabel="Community vaccination coverage (%)",
        ylabel="Vaccination start",
        xticks=np.arange(len(coverage)), xticklabels=[f"{int(c*100)}" for c in coverage],
        yticks=np.arange(len(delay_values)), yticklabels=["Declaration", "+14 days", "+28 days"],
    )
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            color = "white" if z[i, j] >= 58 else "#1F2937"
            axes[1].text(j, i, f"{z[i,j]:.0f}", ha="center", va="center", fontsize=8, color=color)
    cb = fig.colorbar(im, ax=axes[1], fraction=.05, pad=.04)
    cb.set_label("Median mortality reduction (%)")

    for label, ax in zip("AB", axes):
        ax.text(-.18, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.grid(axis="x", color="#E5E7EB")

    fig.tight_layout()
    fig.savefig(REVIEW / "Figure_4_implementation_review.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(REVIEW / "Figure_4_review.pdf", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
