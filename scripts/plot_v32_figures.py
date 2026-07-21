import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter


OUT_DIR = Path("figures/v32")
MASTER_CANDIDATES = [
    Path("data_and_results/final_results_all_scenarios_compiled.csv"),
    Path("data_and_results/final_high_replicate_raw.csv"),
]
SPAGHETTI_DIR = Path("data_and_results/new_spaghetti_chunks")
BASE_OPS_CHUNK_DIR = Path("data_and_results/base_ops_chunks")
CONTOUR_CANDIDATES = [
    Path("data_and_results/fig5_raw_averted_mortality.csv"),
    Path("data_and_results/contour_data.csv"),
]
RISK_CANDIDATES = [
    Path("data_and_results/fig8_raw_averted_mortality.csv"),
    Path("data_and_results/risk_comp_data.csv"),
]

COLORS = {
    "base": "#4A5561",
    "enh": "#4A5561",
    "ring": "#2F6F9F",
    "community": "#1F8E83",
    "hybrid": "#C96549",
    "grid": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "zero": "#2B3037",
}

BENEFIT_CMAP = LinearSegmentedColormap.from_list(
    "benefit",
    ["#F7F4EA", "#E7E4CB", "#CDBD72", "#8FA66B", "#3F8378", "#245B70"],
    N=256,
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging",
    ["#9E4B3F", "#D17A5F", "#E9C39A", "#F7F4EA", "#B8C9B2", "#60908D", "#285A6E"],
    N=256,
)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 8.0,
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
    print(png)
    print(pdf)


def panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
    )


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def load_master():
    usecols = ["scenario", "level", "seed", "population_size", "deaths_percent"]
    keep_scenarios = {
        "analysis_1_reactive_ring",
        "analysis_2_community_vax",
        "analysis_9_contour_det_trace_r2",
    }

    def read_filtered(path):
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250000, low_memory=False):
            sub = chunk[chunk["scenario"].isin(keep_scenarios)]
            if not sub.empty:
                parts.append(sub)
        if not parts:
            return pd.DataFrame(columns=usecols)
        return pd.concat(parts, ignore_index=True)

    path = first_existing(MASTER_CANDIDATES)
    frames = []
    if path is not None:
        frames.append(read_filtered(path))
    else:
        raise FileNotFoundError(
            "Missing data_and_results/final_results_all_scenarios_compiled.csv "
            "or data_and_results/final_high_replicate_raw.csv."
        )

    base_chunks = sorted(BASE_OPS_CHUNK_DIR.glob("base_ops_results_*.csv"))
    frames.extend(read_filtered(path) for path in base_chunks)

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["scenario", "level", "seed"], keep="last")


def add_counts(raw):
    raw = raw.copy()
    raw["population_size"] = pd.to_numeric(raw["population_size"], errors="coerce")
    raw["deaths_percent"] = pd.to_numeric(raw["deaths_percent"], errors="coerce")
    raw["deaths_count"] = raw["deaths_percent"] * raw["population_size"] / 100.0
    if "cases_percent" in raw.columns:
        raw["cases_count"] = raw["cases_percent"] * raw["population_size"] / 100.0
    return raw


def select_level(raw, level, scenario=None):
    sub = raw[raw["level"] == level].copy()
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario].copy()
    if sub.empty:
        raise KeyError(f"Missing level: {scenario or '*'} / {level}")
    return sub


def paired_percent_averted(raw, base, intervention, outcome="deaths_percent"):
    cols = ["seed", outcome]
    base = base.drop_duplicates("seed")
    intervention = intervention.drop_duplicates("seed")
    merged = intervention[cols].merge(
        base[cols].rename(columns={outcome: "baseline_value"}),
        on="seed",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No paired seeds found for requested comparison.")
        
    merged["psa_sample"] = merged["seed"] % 1000
    expected = merged.groupby("psa_sample", observed=False)[["baseline_value", outcome]].mean()
    
    values = np.where(
        expected["baseline_value"] > 0,
        (expected["baseline_value"] - expected[outcome]) / expected["baseline_value"] * 100.0,
        np.nan,
    )
    return values[np.isfinite(values)]


def find_community_level(raw, coverage):
    preferred_scenario = "analysis_2_community_vax"
    preferred_level = f"comm_vax_{coverage}"
    if ((raw["scenario"] == preferred_scenario) & (raw["level"] == preferred_level)).any():
        return preferred_scenario, preferred_level

    candidates = [
        f"base_comm{coverage}",
        f"base_comm_{coverage}",
        f"base_comm_vax_{coverage}",
        f"comm_base_{coverage}",
        f"comm_vax_base_{coverage}",
        f"community_base_{coverage}",
        f"base_community_{coverage}",
    ]
    for level in candidates:
        if (raw["level"] == level).any():
            return None, level

    comm = raw[
        raw["level"].str.contains("comm|community", case=False, na=False)
        & raw["level"].str.contains(str(coverage), na=False)
    ].copy()
    if not comm.empty:
        row = comm[["scenario", "level"]].drop_duplicates().iloc[0]
        return row["scenario"], row["level"]
    raise KeyError(
        f"Missing community vaccination level for {coverage}% coverage."
    )


def plot_fig2_forest(raw):
    raw = add_counts(raw)
    base = select_level(raw, "no_vax_base_ops", "analysis_1_reactive_ring")
    enh = select_level(raw, "no_vax_enh_ops", "analysis_1_reactive_ring")
    enh_ring = select_level(raw, "vax_enh_ops", "analysis_1_reactive_ring")

    rows = [
        ("Enhanced operations", "vs base operations", "enh", paired_percent_averted(raw, base, enh)),
        ("Enhanced operations + ring", "vs base operations", "ring", paired_percent_averted(raw, base, enh_ring)),
        ("Ring vaccination increment", "vs enhanced operations", "ring", paired_percent_averted(raw, enh, enh_ring)),
    ]
    for coverage in [20, 40, 60, 80]:
        scenario, level = find_community_level(raw, coverage)
        community = select_level(raw, level, scenario)
        rows.append(
            (
                f"Community vaccination {coverage}%",
                "vs base operations",
                "community",
                paired_percent_averted(raw, base, community),
            )
        )

    stats = []
    for label, sublabel, color_key, values in rows:
        stats.append(
            {
                "label": label,
                "sublabel": sublabel,
                "color": COLORS[color_key],
                "median": np.nanmedian(values),
                "p25": np.nanpercentile(values, 25),
                "p75": np.nanpercentile(values, 75),
            }
        )

    y = np.arange(len(stats))[::-1]
    fig = plt.figure(figsize=(7.8, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 3.45, 1.42], wspace=0.04)
    label_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=label_ax)
    num_ax = fig.add_subplot(gs[0, 2], sharey=label_ax)

    ax.axvline(0, color=COLORS["zero"], lw=0.9)
    for yi, row in zip(y, stats):
        left_bound = row["p25"]
        if left_bound < -35:
            ax.plot([-35, row["p75"]], [yi, yi], color=COLORS["text"], lw=1.2, solid_capstyle="round")
            ax.plot(-34.0, yi, marker='<', color=COLORS["text"], ms=5.0, clip_on=False)
        else:
            ax.plot([row["p25"], row["p75"]], [yi, yi], color=COLORS["text"], lw=1.2, solid_capstyle="round")
        ax.plot(row["median"], yi, "s", ms=4.0, color=row["color"], mec=COLORS["text"], mew=0.3)
        num_ax.text(
            0.0,
            yi,
            f"{row['median']:.0f}% ({row['p25']:.0f} to {row['p75']:.0f})",
            ha="left",
            va="center",
            fontsize=8.0,
            color=COLORS["text"],
        )
        label_ax.text(0.0, yi + 0.14, row["label"], ha="left", va="center", fontsize=8.6, fontweight="bold", color=COLORS["text"])
        label_ax.text(0.0, yi - 0.18, row["sublabel"], ha="left", va="center", fontsize=7.4, color=COLORS["muted"])

    for sep in [len(stats) - 1.5, len(stats) - 3.5]:
        for axis in [label_ax, ax, num_ax]:
            axis.axhline(sep, color=COLORS["grid"], lw=0.8)

    for axis in [label_ax, ax, num_ax]:
        axis.set_ylim(-0.8, len(stats) - 0.2)
        axis.set_yticks([])

    ax.set_xlim(-35, 100)
    ax.set_xticks([-25, 0, 25, 50, 75, 100])
    ax.set_xlabel("Cumulative deaths averted (% of comparator)")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    label_ax.set_xlim(0, 1)
    num_ax.set_xlim(0, 1)
    for axis in [label_ax, num_ax]:
        axis.set_xticks([])
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    num_ax.text(
        0.0,
        len(stats) - 0.05,
        "Median (95% UI)",
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
    )
    save(fig, "fig2_forest")


def load_spaghetti():
    paths = sorted(SPAGHETTI_DIR.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"Missing spaghetti chunks in {SPAGHETTI_DIR}")
    arrays = {}
    for path in paths:
        z = np.load(path, allow_pickle=True)
        for key in z.files:
            arr = np.asarray(z[key], dtype=float)
            arrays.setdefault(key, []).append(arr)
    return {key: np.vstack(parts) for key, parts in arrays.items()}


def plot_fig3_spaghetti():
    data = load_spaghetti()
    enh_final = np.cumsum(data["enh_no_vax"], axis=1)[:, -1]
    ring_final = np.cumsum(data["enh_ring"], axis=1)[:, -1]
    if np.nanmedian(ring_final) > np.nanmedian(enh_final):
        raise RuntimeError(
            "Existing new_spaghetti_chunks are inconsistent: enh_ring has higher median deaths "
            "than enh_no_vax. Regenerate chunks with scripts/generate_new_spaghetti.py after "
            "the ring_radius=2 fix."
        )
    base = np.cumsum(data["base_no_vax"], axis=1)
    panels = [
        ("enh_no_vax", "Enhanced operations", COLORS["enh"]),
        ("enh_ring", "Enhanced operations + ring 2", COLORS["ring"]),
        ("base_comm40", "Community vaccination 40%", COLORS["community"]),
    ]
    max_plot_day = 90
    days = np.arange(base.shape[1])
    keep_days = days <= max_plot_day
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), sharey=True, gridspec_kw={"wspace": 0.22})
    rng = np.random.default_rng(20260705)
    for ax, (key, title, color), letter in zip(axes, panels, ["A", "B", "C"]):
        intervention = np.cumsum(data[key], axis=1)
        denom = base[:, [-1]]
        averted = np.full_like(base, np.nan, dtype=float)
        np.divide(base - intervention, denom, out=averted, where=denom > 0)
        averted *= 100.0
        final_values = averted[:, -1]
        q25, q75 = np.nanpercentile(final_values, [25, 75])
        iqr_idx = np.flatnonzero((final_values >= q25) & (final_values <= q75))
        if len(iqr_idx) == 0:
            iqr_idx = np.flatnonzero(np.isfinite(final_values))
        idx = rng.choice(iqr_idx, size=min(300, len(iqr_idx)), replace=False)
        for row in averted[idx]:
            ax.plot(days[keep_days], row[keep_days], color=color, alpha=0.045, lw=0.7)
        med = np.nanmedian(averted, axis=0)
        ax.plot(days[keep_days], med[keep_days], color=color, lw=2.3)
        ax.axhline(0, color=COLORS["zero"], lw=0.8, ls="--")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.0)
        ax.set_xlabel("Days since outbreak declaration")
        ax.set_ylim(-5, 100)
        ax.set_xlim(0, max_plot_day)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        panel_label(ax, letter, x=-0.16, y=1.02)
    axes[0].set_ylabel("Cumulative deaths averted (% of baseline)")
    save(fig, "fig3_spaghetti_base_comparator")


def read_matrix(path, index_col=False):
    if index_col:
        df = pd.read_csv(path, index_col=0)
        y = df.index.to_numpy(dtype=float)
        x = df.columns.astype(float).to_numpy()
        z = df.to_numpy(dtype=float)
    else:
        df = pd.read_csv(path)
        y = df["Vaccine_Efficacy_Pct"].to_numpy(dtype=float)
        x = np.array([float(c) for c in df.columns if c != "Vaccine_Efficacy_Pct"], dtype=float)
        z = df.drop(columns=["Vaccine_Efficacy_Pct"]).to_numpy(dtype=float)
    return x, y, z


def plot_matrix_contour(ax, x, y, z, levels, cmap, cbar_label, diverging=False, sigma=0.75):
    z = gaussian_filter(z.astype(float), sigma=sigma)
    X, Y = np.meshgrid(x, y)
    if diverging:
        if levels is None:
            vmax = max(60, 5 * np.ceil(np.nanmax(np.abs(z)) / 5))
            levels = np.arange(-vmax, vmax + 5, 5)
        vmin = float(np.nanmin(levels))
        vmax = float(np.nanmax(levels))
        cf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), extend="both")
        line_levels = [v for v in [-40, -20, 0, 20, 40, 60, 80] if vmin <= v <= vmax]
    else:
        cf = ax.contourf(X, Y, z, levels=levels, cmap=cmap, extend="max")
        line_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    cs = ax.contour(X, Y, z, levels=line_levels, colors=COLORS["text"], linewidths=0.65, alpha=0.82)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f}%", fontsize=6.2)
    ax.spines[["top", "right"]].set_visible(False)
    return cf, cbar_label


def plot_operational_increment_from_master(ax, raw):
    grid = raw[raw["scenario"] == "analysis_9_contour_det_trace_r2"].copy()
    if grid.empty:
        raise KeyError("Missing scenario analysis_9_contour_det_trace_r2")
    grid["det"] = grid["level"].str.extract(r"det_([0-9.]+)").astype(float)
    grid["trace"] = grid["level"].str.extract(r"trace_([0-9.]+)").astype(float)
    grid["is_vax"] = grid["level"].str.endswith("_vax") & ~grid["level"].str.endswith("_no_vax")
    base = grid[~grid["is_vax"]][["seed", "det", "trace", "deaths_percent"]].rename(columns={"deaths_percent": "base_deaths"})
    vax = grid[grid["is_vax"]][["seed", "det", "trace", "deaths_percent"]]
    merged = vax.merge(base, on=["seed", "det", "trace"], how="inner")
    merged["reduction"] = np.where(
        merged["base_deaths"] > 0,
        (merged["base_deaths"] - merged["deaths_percent"]) / merged["base_deaths"] * 100.0,
        np.nan,
    )
    summ = merged.groupby(["trace", "det"], observed=False)["reduction"].median().reset_index()
    pivot = summ.pivot(index="trace", columns="det", values="reduction")
    return pivot.columns.to_numpy(float) * 100, pivot.index.to_numpy(float) * 100, pivot.to_numpy(float)


def plot_operational_vs_base_from_master(ax, raw):
    grid = raw[raw["scenario"] == "analysis_9_contour_det_trace_r2"].copy()
    if grid.empty:
        raise KeyError("Missing scenario analysis_9_contour_det_trace_r2")
    grid["det"] = grid["level"].str.extract(r"det_([0-9.]+)").astype(float)
    grid["trace"] = grid["level"].str.extract(r"trace_([0-9.]+)").astype(float)
    grid["is_vax"] = grid["level"].str.endswith("_vax") & ~grid["level"].str.endswith("_no_vax")
    no_vax = grid[~grid["is_vax"]][["seed", "det", "trace", "deaths_percent"]].copy()

    base = raw[
        (raw["scenario"] == "analysis_1_reactive_ring")
        & (raw["level"] == "no_vax_base_ops")
    ][["seed", "deaths_percent"]].drop_duplicates("seed")
    if base.empty:
        raise KeyError("Missing analysis_1_reactive_ring / no_vax_base_ops")

    merged = no_vax.merge(
        base.rename(columns={"deaths_percent": "base_deaths"}),
        on="seed",
        how="inner",
    )
    if merged.empty:
        base_median = pd.to_numeric(base["deaths_percent"], errors="coerce").median()
        no_vax["reduction"] = (base_median - no_vax["deaths_percent"]) / base_median * 100.0
        summ = no_vax.groupby(["trace", "det"], observed=False)["reduction"].median().reset_index()
    else:
        merged["reduction"] = np.where(
            merged["base_deaths"] > 0,
            (merged["base_deaths"] - merged["deaths_percent"]) / merged["base_deaths"] * 100.0,
            np.nan,
        )
        summ = merged.groupby(["trace", "det"], observed=False)["reduction"].median().reset_index()
    pivot = summ.pivot(index="trace", columns="det", values="reduction")
    return pivot.columns.to_numpy(float) * 100, pivot.index.to_numpy(float) * 100, pivot.to_numpy(float)


def plot_fig4_contours(raw):
    contour_path = first_existing(CONTOUR_CANDIDATES)
    risk_path = first_existing(RISK_CANDIDATES)
    if contour_path is None or risk_path is None:
        raise FileNotFoundError("Missing contour/risk-compensation input files in data_and_results/.")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.65), gridspec_kw={"wspace": 0.34})

    x, y, z = plot_operational_vs_base_from_master(axes[0], raw)
    cf1, label1 = plot_matrix_contour(axes[0], x, y, z, np.arange(-40, 95, 5), DIVERGING_CMAP, "Median mortality reduction (%)", diverging=True, sigma=0.9)
    axes[0].set_xlabel("Index-case detection (%)")
    axes[0].set_ylabel("Contact tracing coverage (%)")
    axes[0].scatter([30], [30], marker="*", s=70, color=COLORS["text"], zorder=5)

    x, y, z = read_matrix(contour_path)
    cf2, label2 = plot_matrix_contour(axes[1], x, y, z, np.arange(0, 75, 5), BENEFIT_CMAP, "Median mortality reduction (%)")
    axes[1].set_xlabel("Community vaccination coverage (%)")
    axes[1].set_ylabel("Vaccine effectiveness (%)")
    axes[1].scatter([40], [45], marker="*", s=70, color=COLORS["text"], zorder=5)

    x, y, z = read_matrix(risk_path)
    cf3, label3 = plot_matrix_contour(axes[2], x, y, z, None, DIVERGING_CMAP, "Median mortality reduction (%)", diverging=True, sigma=1.75)
    axes[2].set_xlabel("Risk-compensation multiplier")
    axes[2].set_ylabel("Vaccine effectiveness (%)")

    for ax, letter in zip(axes, ["A", "B", "C"]):
        panel_label(ax, letter, x=-0.16, y=1.02)
    for ax, cf, label in zip(axes, [cf1, cf2, cf3], [label1, label2, label3]):
        cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.17, fraction=0.08, aspect=25)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=6.7)
    save(fig, "fig4_contours")


def load_fig5_data():
    usecols = ["scenario", "level", "seed", "population_size", "deaths_percent", "vaccines"]
    keep_scenarios = {
        "analysis_1_reactive_ring",
        "analysis_3_hybrid",
        "analysis_4_timing",
        "analysis_5_dose_efficiency",
        "analysis_2_community_vax",
        "fig5_tornado_delay",
        "fig5_tornado_det",
        "fig5_tornado_eff",
        "fig5_tornado_immune",
        "fig5_tornado_trace",
        "fig5_tornado_uptake",
    }

    def read_filtered(path):
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250000, low_memory=False):
            sub = chunk[chunk["scenario"].isin(keep_scenarios)]
            if not sub.empty:
                parts.append(sub)
        if not parts:
            return pd.DataFrame(columns=usecols)
        return pd.concat(parts, ignore_index=True)

    frames = []
    path = first_existing(MASTER_CANDIDATES)
    if path is not None:
        frames.append(read_filtered(path))
    frames.extend(read_filtered(path) for path in sorted(BASE_OPS_CHUNK_DIR.glob("base_ops_results_*.csv")))
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    data = data.drop_duplicates(["scenario", "level", "seed"], keep="last")
    data["population_size"] = pd.to_numeric(data["population_size"], errors="coerce")
    data["deaths_percent"] = pd.to_numeric(data["deaths_percent"], errors="coerce")
    data["vaccines"] = pd.to_numeric(data["vaccines"], errors="coerce")
    data["deaths_count"] = data["deaths_percent"] * data["population_size"] / 100.0
    data["vaccines_per_100k"] = data["vaccines"] / data["population_size"] * 100000.0
    return data


def reduction_values(base, intervention):
    base = base[["seed", "deaths_percent"]].drop_duplicates("seed")
    intervention = intervention[["seed", "deaths_percent"]].drop_duplicates("seed")
    merged = intervention.merge(base.rename(columns={"deaths_percent": "base_deaths"}), on="seed", how="inner")
    if merged.empty:
        intervention["psa_sample"] = intervention["seed"] % 1000
        expected = intervention.groupby("psa_sample", observed=False)["deaths_percent"].mean()
        base_med = base["deaths_percent"].mean()
        values = (base_med - expected) / base_med * 100.0
    else:
        merged["psa_sample"] = merged["seed"] % 1000
        expected = merged.groupby("psa_sample", observed=False)[["base_deaths", "deaths_percent"]].mean()
        values = np.where(
            expected["base_deaths"] > 0,
            (expected["base_deaths"] - expected["deaths_percent"]) / expected["base_deaths"] * 100.0,
            np.nan,
        )
    return np.asarray(values, dtype=float)


def plot_fig5_timing_dose():
    data = load_fig5_data()
    base = data[(data["scenario"] == "analysis_1_reactive_ring") & (data["level"] == "no_vax_base_ops")]
    enh = data[(data["scenario"] == "analysis_1_reactive_ring") & (data["level"] == "no_vax_enh_ops")]
    if base.empty or enh.empty:
        raise KeyError("Missing no-vaccination baselines for Figure 5")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.15),
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.0], "wspace": 0.42},
    )

    timing_specs = [
        ("reactive_detect_delay_0", "At declaration\n(day 0)", "#1F8E83"),
        ("reactive_detect_delay_7", "Declaration\n+7 days", "#CDBD72"),
        ("reactive_detect_delay_14", "Declaration\n+14 days", "#C96549"),
    ]
    timing_rows = []
    for level, label, color in timing_specs:
        sub = data[(data["scenario"] == "analysis_4_timing") & (data["level"] == level)]
        vals = reduction_values(base, sub)
        timing_rows.append((label, color, np.nanmedian(vals)))

    y = np.arange(len(timing_rows))
    ax_a.barh(y, [r[2] for r in timing_rows], color=[r[1] for r in timing_rows], height=0.56, alpha=0.96)
    for yi, (label, color, value) in zip(y, timing_rows):
        ax_a.text(value + 1.8, yi, f"{value:.0f}%", ha="left", va="center", color=color, fontsize=8, fontweight="bold")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([r[0] for r in timing_rows])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Median mortality reduction (%)")
    ax_a.set_xlim(0, 95)
    ax_a.set_xticks([0, 20, 40, 60, 80])
    ax_a.grid(axis="x", color=COLORS["grid"], lw=0.7)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    panel_label(ax_a, "A", x=-0.18, y=1.02)
    ax_a.text(0.0, 1.035, "50% community coverage", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=7.2, color=COLORS["muted"])

    def median_reduction(scenario, level, comparator=enh):
        sub = data[(data["scenario"] == scenario) & (data["level"] == level)]
        if sub.empty:
            return np.nan
        return float(np.nanmedian(reduction_values(comparator, sub)))

    days = np.linspace(0, 30, 301)
    emax = 45.0
    onset_specs = [
        ("Fast onset", 5.0, 1.0, "#C96549"),
        ("Standard onset", 10.0, 0.5, COLORS["community"]),
        ("Slow onset", 14.0, 0.3, COLORS["ring"]),
    ]
    for label, d0, k, color in onset_specs:
        efficacy = emax / (1.0 + np.exp(-k * (days - d0)))
        ax_b.plot(days, efficacy, color=color, lw=1.8, label=label)
    ax_b.plot([0, 10, 10, 30], [0, 0, emax, emax], color=COLORS["muted"], lw=1.25, ls="--", label="Step function")
    ax_b.set_xlabel("Days since vaccination")
    ax_b.set_ylabel("Protection (%)")
    ax_b.set_xlim(0, 30)
    ax_b.set_ylim(0, 50)
    ax_b.set_xticks([0, 10, 20, 30])
    ax_b.grid(color=COLORS["grid"], lw=0.7)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(frameon=False, loc="lower right", fontsize=6.2)
    panel_label(ax_b, "B", x=-0.16, y=1.02)

    onset_rows = [
        (5, "5 d", median_reduction("fig5_tornado_immune", "vax_immune_5.0")),
        (10, "10 d", median_reduction("analysis_1_reactive_ring", "vax_enh_ops")),
        (14, "14 d", median_reduction("fig5_tornado_immune", "vax_immune_14.0")),
    ]
    onset_df = pd.DataFrame(onset_rows, columns=["days", "label", "reduction"]).dropna()
    ax_c.plot(onset_df["days"], onset_df["reduction"], color=COLORS["ring"], lw=1.9, marker="o", ms=4.8)
    for row in onset_df.itertuples(index=False):
        ax_c.text(row.days, row.reduction + 1.2, f"{row.reduction:.0f}%", ha="center", va="bottom", fontsize=7.2, color=COLORS["ring"], fontweight="bold")
    ax_c.scatter([10], [median_reduction("analysis_1_reactive_ring", "vax_enh_ops")], marker="*", s=80, color=COLORS["text"], zorder=5)
    ax_c.text(0.03, 0.94, "Ring 2 vaccination", transform=ax_c.transAxes, color=COLORS["ring"], ha="left", va="top", fontsize=7.6, fontweight="bold")
    ax_c.set_xlabel("Immune-onset midpoint (days)")
    ax_c.set_ylabel("Incremental reduction\nvs enhanced ops (%)")
    ax_c.set_xlim(4, 15)
    ax_c.set_ylim(0, max(30, np.nanmax(onset_df["reduction"]) + 6 if not onset_df.empty else 30))
    ax_c.set_xticks([5, 10, 14])
    ax_c.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax_c.spines[["top", "right"]].set_visible(False)
    panel_label(ax_c, "C", x=-0.16, y=1.02)
    save(fig, "fig5_timing_dose")


def main():
    setup_style()
    raw = load_master()
    tasks = [
        ("Fig 2 forest", lambda: plot_fig2_forest(raw)),
        ("Fig 3 spaghetti", plot_fig3_spaghetti),
        ("Fig 4 contours", lambda: plot_fig4_contours(raw)),
        ("Fig 5 timing/dose", plot_fig5_timing_dose),
    ]
    failed = []
    for label, func in tasks:
        try:
            func()
        except Exception as exc:
            failed.append((label, exc))
            print(f"SKIPPED {label}: {exc}")
    if failed:
        print("\nUnfinished figures:")
        for label, exc in failed:
            print(f"- {label}: {exc}")


if __name__ == "__main__":
    main()
