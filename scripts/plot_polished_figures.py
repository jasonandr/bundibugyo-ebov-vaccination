import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter


OUT_DIR = Path("figures/polished")
RAW_PATH = Path("data_and_results/final_high_replicate_raw.csv")
CONTOUR_CANDIDATES = [Path("data_and_results/fig5_raw_averted_mortality.csv")]
RISK_COMP_CANDIDATES = [Path("data_and_results/fig8_raw_averted_mortality.csv")]
DELIVERY_CANDIDATES = [Path("data_and_results/fig5_delivery_window_v2_raw.csv")]

COLORS = {
    "grey": "#4A5561",
    "ring": "#536F79",
    "community": "#1F8E83",
    "hybrid": "#C96549",
    "amber": "#E8A12C",
    "red": "#B95742",
    "light_grey": "#D1D5DB",
    "grid": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
}

BENEFIT_CMAP = LinearSegmentedColormap.from_list(
    "lancet_benefit",
    ["#F7F4EA", "#E7E4CB", "#CDBD72", "#8FA66B", "#3F8378", "#245B70"],
    N=256,
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "lancet_diverging",
    ["#9E4B3F", "#D17A5F", "#E9C39A", "#F7F4EA", "#B8C9B2", "#60908D", "#285A6E"],
    N=256,
)

STRATEGY_SPECS = [
    ("analysis_1_reactive_ring", "no_vax_enh_ops", "Enhanced ops\nno vaccine", "grey"),
    ("analysis_1_reactive_ring", "vax_enh_ops", "Reactive\nring", "ring"),
    ("analysis_10_comm_vax_base_ops", "comm_vax_20", "Community\n20%", "community"),
    ("analysis_10_comm_vax_base_ops", "comm_vax_40", "Community\n40%", "community"),
    ("analysis_10_comm_vax_base_ops", "comm_vax_60", "Community\n60%", "community"),
    ("analysis_10_comm_vax_base_ops", "comm_vax_80", "Community\n80%", "community"),
    ("analysis_11_hybrid_base_ops", "hybrid_40", "Hybrid\n40% + ring", "hybrid"),
]


def setup_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{stem}={png}")
    print(f"{stem}_pdf={pdf}")


def first_existing(paths, label):
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {label}; checked {checked}")


def panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
    )


def load_raw():
    raw = pd.read_csv(RAW_PATH)
    raw["deaths_count"] = raw["deaths_percent"] * raw["population_size"] / 100.0
    raw["cases_count"] = raw["cases_percent"] * raw["population_size"] / 100.0
    return raw


def level_df(raw, scenario, level):
    sub = raw[(raw["scenario"] == scenario) & (raw["level"] == level)].copy()
    return sub


def paired_reduction(df, baseline, label):
    cols = ["seed", "deaths_count", "cases_count"]
    merged = df.merge(
        baseline[cols].rename(columns={"deaths_count": "baseline_deaths", "cases_count": "baseline_cases"}),
        on="seed",
        how="inner",
    )
    merged["strategy"] = label
    merged["mortality_reduction"] = np.where(
        merged["baseline_deaths"] > 0,
        (merged["baseline_deaths"] - merged["deaths_count"]) / merged["baseline_deaths"] * 100.0,
        np.nan,
    )
    merged["case_reduction"] = np.where(
        merged["baseline_cases"] > 0,
        (merged["baseline_cases"] - merged["cases_count"]) / merged["baseline_cases"] * 100.0,
        np.nan,
    )
    return merged


def median_percent_reduction(value, baseline_value):
    if baseline_value <= 0:
        return np.nan
    return (baseline_value - value) / baseline_value * 100.0


def summary(df, group_cols, value="mortality_reduction"):
    return (
        df.groupby(group_cols, observed=False)[value]
        .agg(median="median", p25=lambda x: np.nanpercentile(x, 25), p75=lambda x: np.nanpercentile(x, 75))
        .reset_index()
    )


def bootstrap_median_reduction(base_values, strategy_values, n_boot=1200, seed=20260704):
    base_values = np.asarray(base_values, dtype=float)
    strategy_values = np.asarray(strategy_values, dtype=float)
    rng = np.random.default_rng(seed)
    n_base = len(base_values)
    n_strategy = len(strategy_values)
    base_med = np.nanmedian(base_values)
    strat_med = np.nanmedian(strategy_values)
    point = median_percent_reduction(strat_med, base_med)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b = np.nanmedian(base_values[rng.integers(0, n_base, n_base)])
        s = np.nanmedian(strategy_values[rng.integers(0, n_strategy, n_strategy)])
        boots[i] = median_percent_reduction(s, b)
    return point, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


def plot_strategy(raw):
    base = level_df(raw, "analysis_1_reactive_ring", "no_vax_base_ops")
    base = base[["seed", "cases_percent", "deaths_percent"]].rename(
        columns={"cases_percent": "base_cases_percent", "deaths_percent": "base_deaths_percent"}
    )
    plot_specs = STRATEGY_SPECS
    order = [s[2] for s in plot_specs]
    rows = []
    for scenario, level, label, color_key in plot_specs:
        sub = level_df(raw, scenario, level)[["seed", "cases_percent", "deaths_percent"]]
        merged = sub.merge(base, on="seed", how="inner")
        for metric, metric_label in [("cases_percent", "Infections"), ("deaths_percent", "Deaths")]:
            base_col = f"base_{metric}"
            reduction = (
                (merged[base_col].to_numpy(dtype=float) - merged[metric].to_numpy(dtype=float))
                / merged[base_col].to_numpy(dtype=float)
                * 100.0
            )
            tmp = pd.DataFrame(
                {
                    "strategy": label,
                    "metric": metric_label,
                    "reduction": reduction,
                    "color_key": color_key,
                }
            )
            rows.append(tmp[np.isfinite(tmp["reduction"])])
    res = pd.concat(rows, ignore_index=True)
    palette = {label: COLORS[color_key] for _, _, label, color_key in plot_specs}

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.1), gridspec_kw={"wspace": 0.12})
    panels = [
        (axes[0], "Infections", "Reduction in infections (%)", "A"),
        (axes[1], "Deaths", "Reduction in deaths (%)", "B"),
    ]
    for ax, metric, xlabel, label in panels:
        sub = res[res["metric"] == metric].copy()
        sns.violinplot(
            data=sub,
            y="strategy",
            x="reduction",
            hue="strategy",
            order=order,
            hue_order=order,
            orient="h",
            palette=palette,
            cut=0,
            inner=None,
            linewidth=0.7,
            width=0.74,
            saturation=0.82,
            legend=False,
            ax=ax,
        )
        for i, strat in enumerate(order):
            vals = sub.loc[sub["strategy"] == strat, "reduction"].to_numpy(dtype=float)
            if len(vals) == 0 or np.isnan(vals).all():
                q25, med, q75 = np.nan, np.nan, np.nan
            else:
                q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
            color = palette[strat]
            ax.plot([q25, q75], [i, i], color=COLORS["text"], lw=1.35, solid_capstyle="round", zorder=3)
            ax.plot(med, i, "o", color="white", mec=COLORS["text"], mew=0.8, ms=4.4, zorder=4)
            ax.text(
                min(med + 3.0, 96),
                i,
                f"{med:.0f}%",
                color=color,
                va="center",
                ha="left",
                fontsize=7.2,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.5},
            )
        ax.axvline(0, color=COLORS["text"], lw=0.8, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.set_xlim(-20, 100)
        ax.set_xticks([-20, 0, 20, 40, 60, 80, 100])
        ax.grid(axis="x", color=COLORS["grid"], lw=0.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, label, x=-0.1 if ax is axes[0] else -0.06, y=1.01)
    axes[1].set_yticklabels([])
    save(fig, "fig_strategy_polished")


def plot_contour():
    df = pd.read_csv(first_existing(CONTOUR_CANDIDATES, "community coverage contour data"))
    y = df["Vaccine_Efficacy_Pct"].to_numpy(dtype=float)
    x = np.array([float(c) for c in df.columns if c != "Vaccine_Efficacy_Pct"], dtype=float)
    z = df.drop(columns=["Vaccine_Efficacy_Pct"]).to_numpy(dtype=float)
    z_smooth = gaussian_filter(z, sigma=0.75)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(5.1, 4.25))
    levels = np.arange(0, 75, 5)
    cf = ax.contourf(X, Y, z_smooth, levels=levels, cmap=BENEFIT_CMAP, extend="max")
    line_levels = [0, 10, 20, 30, 40, 50, 60]
    cs = ax.contour(X, Y, z_smooth, levels=line_levels, colors=COLORS["text"], linewidths=0.75, alpha=0.85)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f}%", fontsize=7)
    zero = ax.contour(X, Y, z_smooth, levels=[0], colors=COLORS["text"], linewidths=1.3)
    ax.clabel(zero, inline=True, fmt={0: "0%"}, fontsize=7)
    ax.scatter([40], [45], marker="*", s=75, color=COLORS["text"], zorder=5)
    ax.annotate(
        "base case\n45% VE",
        xy=(40, 45),
        xytext=(31.5, 53.5),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["text"]},
        fontsize=6.8,
        color=COLORS["text"],
        ha="left",
        va="center",
    )
    ax.set_xlabel("Community vaccination coverage (%)")
    ax.set_ylabel("Vaccine effectiveness (VE, %)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(cf, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label("Median mortality reduction (%)")
    cbar.ax.tick_params(labelsize=7)
    save(fig, "fig_contour_polished")


def _plot_coverage_effect_contour(ax):
    df = pd.read_csv(first_existing(CONTOUR_CANDIDATES, "community coverage contour data"))
    y = df["Vaccine_Efficacy_Pct"].to_numpy(dtype=float)
    x = np.array([float(c) for c in df.columns if c != "Vaccine_Efficacy_Pct"], dtype=float)
    z = df.drop(columns=["Vaccine_Efficacy_Pct"]).to_numpy(dtype=float)
    z_smooth = gaussian_filter(z, sigma=0.75)
    X, Y = np.meshgrid(x, y)

    cf = ax.contourf(X, Y, z_smooth, levels=np.arange(0, 75, 5), cmap=BENEFIT_CMAP, extend="max")
    cs = ax.contour(X, Y, z_smooth, levels=[10, 20, 30, 40, 50, 60], colors=COLORS["text"], linewidths=0.65, alpha=0.8)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f}%", fontsize=6.3)
    ax.scatter([40], [45], marker="*", s=65, color=COLORS["text"], zorder=5)
    ax.annotate(
        "base case\n45% VE",
        xy=(40, 45),
        xytext=(27, 55),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": COLORS["text"]},
        fontsize=6.4,
        color=COLORS["text"],
        ha="left",
        va="center",
    )
    ax.set_xlabel("Community vaccination coverage (%)")
    ax.set_ylabel("Vaccine effectiveness (VE, %)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.spines[["top", "right"]].set_visible(False)
    return cf


def _plot_risk_comp_contour(ax):
    df = pd.read_csv(first_existing(RISK_COMP_CANDIDATES, "risk-compensation contour data"), index_col=0)
    y = df.index.to_numpy(dtype=float)
    x = df.columns.astype(float).to_numpy()
    z = df.to_numpy(dtype=float)
    z_smooth = gaussian_filter(z, sigma=1.2)
    X, Y = np.meshgrid(x, y)
    vmax = max(60, 5 * np.ceil(np.nanmax(np.abs(z_smooth)) / 5))
    levels = np.arange(-vmax, vmax + 5, 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(X, Y, z_smooth, levels=levels, cmap=DIVERGING_CMAP, norm=norm, extend="both")
    line_levels = [level for level in [-40, -20, -10, 0, 20, 40, 60] if -vmax <= level <= vmax]
    cs = ax.contour(X, Y, z_smooth, levels=line_levels, colors=COLORS["text"], linewidths=0.75, alpha=0.82)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f}%", fontsize=6.3)
    ax.contour(X, Y, z_smooth, levels=[0], colors=COLORS["text"], linewidths=1.25)
    ax.set_xlabel("Risk-compensation multiplier")
    ax.set_ylabel("Vaccine effectiveness (VE, %)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.spines[["top", "right"]].set_visible(False)
    return cf


def _plot_det_trace_contour_r2(ax):
    raw = load_raw()
    # Filter for the new grid
    grid = raw[raw["scenario"] == "analysis_9_contour_det_trace_r2"].copy()
    
    # Extract det and trace parameters from the level name
    import re
    # We expect levels like "det_0.1_trace_0.5_no_vax" or "det_0.1_trace_0.5_vax"
    grid["det"] = grid["level"].apply(lambda x: float(re.search(r'det_([0-9.]+)', x).group(1)))
    grid["trace"] = grid["level"].apply(lambda x: float(re.search(r'trace_([0-9.]+)', x).group(1)))
    grid["is_vax"] = grid["level"].apply(lambda x: "no_vax" not in x)
    
    # Calculate median deaths per configuration
    summ = grid.groupby(["det", "trace", "is_vax"], observed=False)["deaths_percent"].median().reset_index()
    
    # We want to compare vax + det + trace against the "status quo" (which we take as baseline no_vax_base_ops)
    baseline = level_df(raw, "analysis_1_reactive_ring", "no_vax_base_ops")
    baseline_deaths = baseline["deaths_percent"].median()
    
    # Filter only the vax scenarios
    vax = summ[summ["is_vax"] == True].copy()
    vax["reduction"] = (baseline_deaths - vax["deaths_percent"]) / baseline_deaths * 100.0
    
    # We want det on x-axis, trace on y-axis
    pivoted = vax.pivot(index="trace", columns="det", values="reduction")
    y = pivoted.index.to_numpy(dtype=float) * 100.0
    x = pivoted.columns.to_numpy(dtype=float) * 100.0
    z = pivoted.to_numpy(dtype=float)
    
    z_smooth = gaussian_filter(z, sigma=0.75)
    X, Y = np.meshgrid(x, y)

    cf = ax.contourf(X, Y, z_smooth, levels=np.arange(0, 105, 5), cmap=BENEFIT_CMAP, extend="max")
    cs = ax.contour(X, Y, z_smooth, levels=[10, 20, 30, 40, 50, 60, 70, 80], colors=COLORS["text"], linewidths=0.65, alpha=0.8)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f}%", fontsize=6.3)
    
    # Add a star for the base case (assuming 80% det, 80% trace was the original 'enh_ops')
    ax.scatter([80], [80], marker="*", s=65, color=COLORS["text"], zorder=5)
    
    ax.set_xlabel("Index-case detection (%)")
    ax.set_ylabel("Contact tracing coverage (%)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.spines[["top", "right"]].set_visible(False)
    return cf


def plot_contour_pair():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.34})
    cf1 = _plot_coverage_effect_contour(axes[0])
    cf2 = _plot_risk_comp_contour(axes[1])
    try:
        cf3 = _plot_det_trace_contour_r2(axes[2])
    except Exception as e:
        print("Could not plot det_trace contour:", e)
        cf3 = cf1
    
    panel_label(axes[0], "A", x=-0.16, y=1.02)
    panel_label(axes[1], "B", x=-0.16, y=1.02)
    panel_label(axes[2], "C", x=-0.16, y=1.02)
    
    cbar1 = fig.colorbar(cf1, ax=axes[0], orientation="horizontal", pad=0.18, fraction=0.08, aspect=28)
    cbar1.set_label("Median mortality reduction (%)")
    cbar1.ax.tick_params(labelsize=6.8)
    
    cbar2 = fig.colorbar(cf2, ax=axes[1], orientation="horizontal", pad=0.18, fraction=0.08, aspect=28)
    cbar2.set_label("Median mortality reduction (%)")
    cbar2.ax.tick_params(labelsize=6.8)
    
    if 'cf3' in locals():
        cbar3 = fig.colorbar(cf3, ax=axes[2], orientation="horizontal", pad=0.18, fraction=0.08, aspect=28)
        cbar3.set_label("Median mortality reduction (%)")
        cbar3.ax.tick_params(labelsize=6.8)
        
    save(fig, "fig_contours_abc")


def plot_dose_frontier(raw):
    frames = []

    dose = raw[raw["scenario"] == "analysis_5_dose_efficiency"].copy()
    for strategy, prefix in [("Reactive ring", "ring_only_uptake"), ("Community vaccination", "comm_only_cov")]:
        sub = dose[dose["level"].str.startswith(prefix)].copy()
        sub["coverage"] = sub["level"].str.extract(r"_(\d+)$").astype(float)
        frames.append((strategy, sub))

    hybrid = raw[raw["scenario"] == "analysis_3_hybrid"].copy()
    hybrid["coverage"] = hybrid["level"].str.extract(r"_(\d+)$").astype(float)
    frames.append(("Hybrid", hybrid))

    paired = []
    for strategy, sub in frames:
        for level, g in sub.groupby("level", observed=False):
            tmp = g.copy()
            tmp["strategy"] = strategy
            tmp["coverage"] = float(g["coverage"].iloc[0])
            tmp["vaccines_per_100k"] = tmp["vaccines"] / tmp["population_size"] * 100000.0
            tmp["deaths_per_100k"] = tmp["deaths_percent"] * 1000.0
            paired.append(tmp)
    df = pd.concat(paired, ignore_index=True)
    summ = (
        df.groupby(["strategy", "coverage"], observed=False)
        .agg(
            x=("vaccines_per_100k", "median"),
            y=("deaths_per_100k", "median"),
        )
        .reset_index()
    )
    baseline = level_df(raw, "analysis_1_reactive_ring", "no_vax_base_ops")
    baseline_deaths = baseline["deaths_percent"].median() * 1000.0
    summ["mortality_reduction"] = (baseline_deaths - summ["y"]) / baseline_deaths * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.35), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.33})
    ax = axes[0]
    ax_ring = axes[1]
    styles = {
        "Reactive ring": COLORS["ring"],
        "Community vaccination": COLORS["community"],
        "Hybrid": COLORS["hybrid"],
    }
    for strategy in ["Community vaccination", "Hybrid"]:
        s = summ[summ["strategy"] == strategy].sort_values("x")
        c = styles[strategy]
        ax.plot(s["x"], s["mortality_reduction"], color=c, lw=1.9, marker="o", ms=4.4)

    ax.text(52000, 69, "Community\nvaccination", color=COLORS["community"], va="center", ha="left", fontsize=7.6, fontweight="bold")
    ax.text(65000, 82, "Hybrid", color=COLORS["hybrid"], va="center", ha="left", fontsize=7.6, fontweight="bold")

    for cov in [20, 40, 60, 80]:
        row = summ[(summ["strategy"] == "Community vaccination") & (summ["coverage"] == cov)]
        if not row.empty:
            r = row.iloc[0]
            ax.text(r["x"], r["mortality_reduction"] - 4.5, f"{int(cov)}%", ha="center", color=COLORS["community"], fontsize=6.9)

    ax.set_xlabel("Vaccine courses delivered per 100,000 population")
    ax.set_ylabel("Median mortality reduction (%)")
    ax.set_xlim(0, 85000)
    ax.set_ylim(0, 92)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ring = summ[summ["strategy"] == "Reactive ring"].sort_values("x")
    ax_ring.plot(ring["x"], ring["mortality_reduction"], color=COLORS["ring"], lw=1.8, marker="o", ms=4.2)
    ax_ring.text(0.03, 0.94, "Reactive ring", transform=ax_ring.transAxes, color=COLORS["ring"], ha="left", va="top", fontsize=7.6, fontweight="bold")
    for cov in [10, 30, 50]:
        row = ring[ring["coverage"] == cov]
        if not row.empty:
            r = row.iloc[0]
            ax_ring.text(r["x"], r["mortality_reduction"] + 0.75, f"{int(cov)}%", ha="center", color=COLORS["ring"], fontsize=6.7)
    ax_ring.set_xlabel("Courses per 100,000")
    ax_ring.set_ylabel("")
    ax_ring.set_xlim(0, 25000)
    ax_ring.set_ylim(0, 92)
    ax_ring.set_xticks([0, 5000, 15000, 25000])
    ax_ring.set_xticklabels(["0", "5k", "15k", "25k"])
    ax_ring.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax_ring.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "A", x=-0.12)
    panel_label(ax_ring, "B", x=-0.16)
    save(fig, "fig_dose_frontier_polished")


def plot_timing(raw):
    baseline = level_df(raw, "analysis_1_reactive_ring", "no_vax_base_ops")
    base_deaths = baseline["deaths_percent"].median() * 1000.0
    specs = [
        ("reactive_detect_delay_0", "Start at\ndeclaration"),
        ("reactive_detect_delay_7", "Start +7\ndays"),
        ("reactive_detect_delay_14", "Start +14\ndays"),
    ]
    rows = []
    for level, label in specs:
        tmp = level_df(raw, "analysis_4_timing", level).copy()
        tmp["strategy"] = label
        tmp["deaths_per_100k"] = tmp["deaths_percent"] * 1000.0
        tmp["mortality_reduction"] = (base_deaths - tmp["deaths_per_100k"]) / base_deaths * 100.0
        rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)
    summ = summary(df, "strategy")
    order = [s[1] for s in specs]
    summ["strategy"] = pd.Categorical(summ["strategy"], categories=order, ordered=True)
    summ = summ.sort_values("strategy")
    y = np.arange(len(summ))
    bar_colors = ["#1F8E83", "#4B9A8C", "#C96549"]

    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.0], hspace=0.58, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax_ring = fig.add_subplot(gs[0, 1])
    ax_comm = fig.add_subplot(gs[1, :])

    ax.barh(y, summ["median"], color=bar_colors, height=0.56, alpha=0.96)
    for yi, row in zip(y, summ.itertuples()):
        label_value = 0.0 if abs(row.median) < 0.5 else row.median
        ax.text(
            row.median + 1.8,
            yi,
            f"{label_value:.0f}%",
            ha="left",
            va="center",
            color=bar_colors[yi],
            fontsize=8,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(summ["strategy"])
    ax.invert_yaxis()
    ax.set_xlabel("Median mortality reduction (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x", color=COLORS["grid"], lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "A", x=-0.18, y=1.02)
    ax.text(0.0, 1.035, "50% community coverage", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color=COLORS["muted"])

    dose = raw[raw["scenario"] == "analysis_5_dose_efficiency"].copy()
    frames = []
    for strategy, prefix, color_key in [
        ("Reactive ring", "ring_only_uptake", "ring"),
        ("Community vaccination", "comm_only_cov", "community"),
    ]:
        sub = dose[dose["level"].str.startswith(prefix)].copy()
        sub["coverage"] = sub["level"].str.extract(r"_(\d+)$").astype(float)
        sub["strategy"] = strategy
        sub["color_key"] = color_key
        frames.append(sub)
    hybrid = raw[raw["scenario"] == "analysis_3_hybrid"].copy()
    hybrid["coverage"] = hybrid["level"].str.extract(r"_(\d+)$").astype(float)
    hybrid["strategy"] = "Hybrid"
    hybrid["color_key"] = "hybrid"
    frames.append(hybrid)

    dose_df = pd.concat(frames, ignore_index=True)
    dose_df["courses_per_100k"] = dose_df["vaccines"] / dose_df["population_size"] * 100000.0
    dose_df["deaths_per_100k"] = dose_df["deaths_percent"] * 1000.0
    dose_summ = (
        dose_df.groupby(["strategy", "coverage", "color_key"], observed=False)
        .agg(courses=("courses_per_100k", "median"), deaths=("deaths_per_100k", "median"))
        .reset_index()
    )
    dose_summ["mortality_reduction"] = (base_deaths - dose_summ["deaths"]) / base_deaths * 100.0

    ring = dose_summ[dose_summ["strategy"] == "Reactive ring"].sort_values("courses")
    ring = ring.assign(mortality_reduction=ring["mortality_reduction"].cummax())
    ax_ring.plot(
        ring["courses"],
        ring["mortality_reduction"],
        color=COLORS["ring"],
        lw=1.9,
        marker="o",
        ms=4.4,
    )
    ax_ring.text(
        0.03,
        0.94,
        "Reactive ring",
        transform=ax_ring.transAxes,
        color=COLORS["ring"],
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
    )
    ax_ring.set_xlabel("Vaccine courses delivered per 100,000")
    ax_ring.set_ylabel("Median mortality reduction (%)")
    ax_ring.set_xlim(0, 25000)
    ax_ring.set_ylim(0, 92)
    ax_ring.set_xticks([0, 5000, 15000, 25000])
    ax_ring.set_xticklabels(["0", "5k", "15k", "25k"])
    ax_ring.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax_ring.spines[["top", "right"]].set_visible(False)
    panel_label(ax_ring, "B", x=-0.16, y=1.02)

    for strategy, color_key, marker in [
        ("Community vaccination", "community", "o"),
        ("Hybrid", "hybrid", "s"),
    ]:
        sub = dose_summ[dose_summ["strategy"] == strategy].sort_values("courses")
        if sub.empty:
            continue
        sub = sub.assign(mortality_reduction=sub["mortality_reduction"].cummax())
        ax_comm.plot(
            sub["courses"],
            sub["mortality_reduction"],
            color=COLORS[color_key],
            lw=1.9,
            marker=marker,
            ms=4.3,
            label=strategy,
        )
    for cov, yoff in [(60, -5.5), (80, -5.5)]:
        row = dose_summ[(dose_summ["strategy"] == "Community vaccination") & (dose_summ["coverage"] == cov)]
        if row.empty:
            continue
        r = row.iloc[0]
        ax_comm.text(
            r["courses"],
            r["mortality_reduction"] + yoff,
            f"{int(cov)}%",
            ha="center",
            va="center",
            color=COLORS["community"],
            fontsize=7.0,
            fontweight="bold",
        )
    ax_comm.set_xlabel("Vaccine courses delivered per 100,000")
    ax_comm.set_ylabel("Median mortality reduction (%)")
    ax_comm.set_xlim(0, 85000)
    ax_comm.set_ylim(0, 92)
    ax_comm.set_xticks([0, 20000, 40000, 60000, 80000])
    ax_comm.set_xticklabels(["0", "20k", "40k", "60k", "80k"])
    ax_comm.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax_comm.spines[["top", "right"]].set_visible(False)
    ax_comm.legend(frameon=False, loc="upper left")
    panel_label(ax_comm, "C", x=-0.08, y=1.02)
    save(fig, "fig_timing_polished")


def plot_delivery_window():
    try:
        delivery_path = first_existing(DELIVERY_CANDIDATES, "delivery-window data")
    except FileNotFoundError:
        return
    data = pd.read_csv(delivery_path, low_memory=False)
    phase = data[data["record_type"] == "phase"].copy()
    milestone = data[data["record_type"] == "milestone"].copy()

    phase_order = ["Before exposure", "Incubation", "After symptoms", "After outcome"]
    phase_colors = {
        "Before exposure": COLORS["community"],
        "Incubation": COLORS["amber"],
        "After symptoms": COLORS["red"],
        "After outcome": COLORS["light_grey"],
    }
    phase_sum = phase.groupby(["strategy", "phase"], observed=False)["percent"].median().reset_index()
    pivot = (
        phase_sum.pivot(index="strategy", columns="phase", values="percent")
        .reindex(["Community vaccination", "Reactive ring"])[phase_order]
        .fillna(0)
    )

    ms = (
        milestone.groupby(["strategy", "milestone"], observed=False)["percent"]
        .agg(median="median", p25=lambda x: np.nanpercentile(x, 25), p75=lambda x: np.nanpercentile(x, 75))
        .reset_index()
    )

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for ph in phase_order:
        vals = pivot[ph].to_numpy(dtype=float)
        ax1.barh(y, vals, left=left, color=phase_colors[ph], edgecolor="white", linewidth=1.0, height=0.56, label=ph)
        for i, v in enumerate(vals):
            if v >= 9:
                txt_color = "white" if ph in ["Before exposure", "After symptoms"] else COLORS["text"]
                ax1.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color=txt_color)
        left += vals
    ax1.set_yticks(y)
    ax1.set_yticklabels(pivot.index)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Timing phase among vaccinated secondary cases (%)")
    ax1.grid(axis="x", color=COLORS["grid"], lw=0.7)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, handlelength=1.0, columnspacing=1.2)
    panel_label(ax1, "A", x=-0.14)

    milestone_order = ["Before exposure", "Before symptoms", "Before death"]
    offsets = {"Community vaccination": -0.11, "Reactive ring": 0.11}
    for strategy, color in [("Community vaccination", COLORS["community"]), ("Reactive ring", COLORS["ring"])]:
        sub = ms[ms["strategy"] == strategy].set_index("milestone").reindex(milestone_order)
        xs = np.arange(len(milestone_order)) + offsets[strategy]
        med = sub["median"].to_numpy(dtype=float)
        p25 = sub["p25"].to_numpy(dtype=float)
        p75 = sub["p75"].to_numpy(dtype=float)
        ax2.errorbar(xs, med, yerr=[med - p25, p75 - med], fmt="o", color=color, ecolor=color, elinewidth=1.8, capsize=3, ms=5.5, label=strategy)
        for x0, m in zip(xs, med):
            ax2.text(x0, m + 2.1, f"{m:.0f}%", ha="center", va="bottom", color=color, fontsize=7.8, fontweight="bold")
    ax2.set_xticks(np.arange(len(milestone_order)))
    ax2.set_xticklabels(["Before\nexposure", "Before\nsymptoms", "Before\ndeath"])
    ax2.set_ylabel("Vaccinated before milestone (%)")
    ax2.set_ylim(0, 55)
    ax2.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")
    panel_label(ax2, "B", x=-0.18)
    save(fig, "fig_delivery_window_polished")


def main():
    setup_style()
    raw = load_raw()
    plot_strategy(raw)
    plot_contour()
    plot_contour_pair()
    plot_dose_frontier(raw)
    plot_timing(raw)
    plot_delivery_window()


if __name__ == "__main__":
    main()
