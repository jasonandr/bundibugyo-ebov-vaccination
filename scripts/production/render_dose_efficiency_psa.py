"""Render a review-only doses, deaths-averted, and efficiency figure.

The input is the matched expected-value PSA produced with --include-ring1.
Each displayed uncertainty interval is the interquartile range across the
parameter-draw expected values, not a stochastic-replicate interval.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGIES = [
    ("ring1_vax_base_ops", "Ring 1 + base operations", "#2878B5"),
    ("vax_base_ops", "Ring 2 + base operations", "#2878B5"),
    ("ring1_vax_enh_ops", "Ring 1 + enhanced operations", "#2878B5"),
    ("vax_enh_ops", "Ring 2 + enhanced operations", "#2878B5"),
    ("comm_base_20", "Community vaccination 20%", "#21918C"),
    ("comm_base_40", "Community vaccination 40%", "#21918C"),
    ("comm_base_60", "Community vaccination 60%", "#21918C"),
    ("comm_base_80", "Community vaccination 80%", "#21918C"),
]


def summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--overwrite-review-artifacts", action="store_true",
                        help="Replace only an incomplete review render from a failed prior attempt.")
    args = parser.parse_args()
    if (args.output.exists() or args.summary_output.exists()) and not args.overwrite_review_artifacts:
        raise FileExistsError("Refusing to overwrite an existing review figure or summary.")

    with args.draw_values.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_strategy = {key: [] for key, _, _ in STRATEGIES}
    for row in rows:
        if row["scenario"] in by_strategy:
            comparator = float(row["comparator_mean_deaths"])
            deaths = float(row["mean_deaths"])
            averted = comparator - deaths
            by_strategy[row["scenario"]].append({
                "doses": float(row["mean_doses"]),
                "deaths_averted": averted,
                "doses_per_death_averted": np.nan if averted <= 0 else float(row["mean_doses"]) / averted,
            })
    missing = [key for key, _, _ in STRATEGIES if not by_strategy[key]]
    if missing:
        raise ValueError(f"Missing expected values for: {missing}")

    output_rows = []
    metrics = ("doses", "deaths_averted", "doses_per_death_averted")
    for key, label, _ in STRATEGIES:
        record = {"scenario": key, "strategy": label, "n_draws": len(by_strategy[key])}
        for metric in metrics:
            vals = [r[metric] for r in by_strategy[key] if np.isfinite(r[metric])]
            med, lo, hi = summary(vals)
            record.update({f"{metric}_median": med, f"{metric}_iqr_low": lo, f"{metric}_iqr_high": hi,
                           f"{metric}_n": len(vals)})
        output_rows.append(record)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.9), sharey=True, gridspec_kw={"wspace": 0.30})
    y = np.arange(len(STRATEGIES))[::-1]
    specifications = [
        ("doses", "Vaccine courses administered", 1000, "Vaccine courses (thousands)"),
        ("deaths_averted", "Deaths averted vs base operations", 1, "Deaths averted"),
        ("doses_per_death_averted", "Vaccine efficiency", 1, "Vaccine courses per death averted"),
    ]
    for panel, (ax, (metric, title, scale, xlabel)) in enumerate(zip(axes, specifications)):
        vals = np.array([r[f"{metric}_median"] / scale for r in output_rows])
        lows = np.array([r[f"{metric}_iqr_low"] / scale for r in output_rows])
        highs = np.array([r[f"{metric}_iqr_high"] / scale for r in output_rows])
        colors = [color for _, _, color in STRATEGIES]
        finite = np.isfinite(vals)
        ax.barh(y[finite], vals[finite], color=np.asarray(colors)[finite], height=0.61, alpha=0.94)
        max_value = np.nanmax(highs)
        axis_low = min(0.0, float(np.nanmin(lows)) * 1.20) if metric == "deaths_averted" else 0.0
        for yi, value in zip(y, vals):
            text = (f"{value:,.1f}" if metric == "doses" else f"{value:,.0f}") if np.isfinite(value) else "Not estimable*"
            if np.isfinite(value) and value < 0:
                x, align = value - max_value * 0.018, "right"
            else:
                # No intervals are displayed: keep each label close to its bar end.
                x, align = (value + max_value * 0.025 if np.isfinite(value) else max_value * 0.018), "left"
            ax.text(x, yi, text, va="center", ha=align, fontsize=9.2, color="#172033")
        ax.set_title(f"{'ABC'[panel]}. {title}", loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.grid(axis="x", color="#E1E4E8", lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(axis_low, max_value * 1.22)
        if axis_low < 0:
            ax.axvline(0, color="#667085", lw=1.0)
    axes[0].set_yticks(y, [label for _, label, _ in STRATEGIES], fontsize=10.3, fontweight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
