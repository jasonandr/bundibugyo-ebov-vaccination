"""Render the dose-efficiency figure (Figure 5).

Improvements over the review render (render_dose_efficiency_psa.py):
  - Panel B honestly labelled "Deaths averted" (comparator detail lives in the
    figure legend, not on the figure).
  - Panel C retitled "Dose efficiency".
  - Optional subtle IQR whiskers (across the 200 parameter-draw expected values).

Input: radius-matched matched expected-value PSA (200 draws x 50 replicates).
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
    parser.add_argument("--whiskers", action="store_true", help="Draw IQR whiskers across parameter draws")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

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
    records = []
    for key, label, _ in STRATEGIES:
        rec = {"label": label}
        for metric in ("doses", "deaths_averted", "doses_per_death_averted"):
            vals = [r[metric] for r in by_strategy[key] if np.isfinite(r[metric])]
            rec[metric] = summary(vals)
        records.append(rec)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.9), sharey=True, gridspec_kw={"wspace": 0.30})
    y = np.arange(len(STRATEGIES))[::-1]
    specifications = [
        ("doses", "Vaccine courses administered", 1000, "Vaccine courses (thousands)"),
        ("deaths_averted", "Deaths averted", 1, "Deaths averted vs matched comparator"),
        ("doses_per_death_averted", "Dose efficiency", 1, "Vaccine courses per death averted"),
    ]
    for panel, (ax, (metric, title, scale, xlabel)) in enumerate(zip(axes, specifications)):
        vals = np.array([r[metric][0] / scale for r in records])
        lows = np.array([r[metric][1] / scale for r in records])
        highs = np.array([r[metric][2] / scale for r in records])
        colors = [color for _, _, color in STRATEGIES]
        finite = np.isfinite(vals)
        ax.barh(y[finite], vals[finite], color=np.asarray(colors)[finite], height=0.61, alpha=0.94)
        if args.whiskers:
            ax.errorbar(vals[finite], y[finite],
                        xerr=np.vstack([vals[finite] - lows[finite], highs[finite] - vals[finite]]),
                        fmt="none", ecolor="#172033", elinewidth=1.1, capsize=2.6, alpha=0.75)
        max_value = np.nanmax(highs)
        label_anchor = highs if args.whiskers else vals
        for yi, value, anchor in zip(y, vals, label_anchor):
            text = (f"{value:,.1f}" if metric == "doses" else f"{value:,.0f}") if np.isfinite(value) else "Not estimable*"
            x, align = (anchor + max_value * 0.025 if np.isfinite(value) else max_value * 0.018), "left"
            ax.text(x, yi, text, va="center", ha=align, fontsize=9.2, color="#172033")
        ax.set_title(f"{'ABC'[panel]}. {title}", loc="left", fontweight="bold", fontsize=11.5)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.grid(axis="x", color="#E1E4E8", lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.0, max_value * (1.30 if args.whiskers else 1.22))
    axes[0].set_yticks(y, [label for _, label, _ in STRATEGIES], fontsize=10.3, fontweight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
