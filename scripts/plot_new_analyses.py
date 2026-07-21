import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT_DIR = "figures/new_analyses"
RAW_PATH = "data_and_results/final_high_replicate_raw.csv"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})

raw = pd.read_csv(RAW_PATH)
raw["deaths_count"] = raw["deaths_percent"] * raw["population_size"] / 100.0
raw["cases_count"] = raw["cases_percent"] * raw["population_size"] / 100.0


def scenario_level(level):
    return raw[(raw["scenario"] == "analysis_1_reactive_ring") & (raw["level"] == level)].copy()


BASE_OPS = scenario_level("no_vax_base_ops")
ENH_OPS = scenario_level("no_vax_enh_ops")


def paired_against(df, baseline, label, comparator_label):
    cols = ["seed", "deaths_count", "cases_count"]
    merged = df.merge(
        baseline[cols].rename(columns={
            "deaths_count": "baseline_deaths",
            "cases_count": "baseline_cases",
        }),
        on="seed",
        how="inner",
    )
    merged["Strategy"] = label
    merged["Comparator"] = comparator_label
    merged["Change in mortality"] = merged["deaths_count"] - merged["baseline_deaths"]
    merged["Cases averted"] = merged["baseline_cases"] - merged["cases_count"]
    merged["Change in mortality (%)"] = np.where(
        merged["baseline_deaths"] > 0,
        merged["Change in mortality"] / merged["baseline_deaths"] * 100.0,
        np.nan,
    )
    return merged


def level_df(scenario, level):
    df = raw[(raw["scenario"] == scenario) & (raw["level"] == level)].copy()
    if df["radius"].nunique() > 1:
        df = df[df["radius"] == df["radius"].max()].copy()
    return df


def add_median_labels(ax, df, x, y, fmt="{:.0f}", dy=1.5):
    for idx, label in enumerate(df[x].dropna().unique()):
        vals = df.loc[df[x] == label, y].dropna()
        if vals.empty:
            continue
        med = float(np.median(vals))
        ax.text(idx, med + dy, fmt.format(med), ha="center", va="bottom", fontsize=10, fontweight="bold")


def summarize_group(df, group_col, value_col):
    return (
        df.groupby(group_col, observed=False)[value_col]
        .agg(
            median="median",
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )


def horizontal_iqr(ax, summary, labels, colors, xlabel, title, annotate=True):
    y = np.arange(len(labels))
    lookup = summary.set_index("Strategy")
    med = np.array([lookup.loc[label, "median"] for label in labels], dtype=float)
    p25 = np.array([lookup.loc[label, "p25"] for label in labels], dtype=float)
    p75 = np.array([lookup.loc[label, "p75"] for label in labels], dtype=float)
    for i, label in enumerate(labels):
        ax.errorbar(
            med[i],
            y[i],
            xerr=[[med[i] - p25[i]], [p75[i] - med[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2,
            capsize=4,
            markersize=6,
        )
        if annotate:
            ax.text(med[i], y[i] + 0.28, f"{med[i]:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.axvline(0, color="black", linewidth=0.8)


def save(fig, name):
    path = f"{OUT_DIR}/{name}_{TIMESTAMP}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}={path}")
    return path


# ---------------------------------------------------------------------
# Extract Radius 2 Ring from spaghetti chunks
import os
all_nv_mod = []
all_r2_mod = []
for i in range(1, 101):
    path = f"data_and_results/spaghetti_chunks/chunk_{i}.npz"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        for a in d['no_vax_mod']:
            all_nv_mod.append(np.sum(a))
        for a in d['ring2_mod']:
            all_r2_mod.append(np.sum(a))

df_r2_ring = pd.DataFrame({
    "seed": np.arange(len(all_r2_mod)),
    "deaths_count": all_r2_mod,
    "cases_count": all_r2_mod # approximation, not used for deaths averted plot
})
df_r2_base = pd.DataFrame({
    "seed": np.arange(len(all_nv_mod)),
    "deaths_count": all_nv_mod,
    "cases_count": all_nv_mod
})
r2_paired = paired_against(df_r2_ring, df_r2_base, "Reactive ring (Radius 2)\nmoderate operations", "Moderate operations")

# ---------------------------------------------------------------------
# Figure 3: Strategy comparison using the correct operational comparator
# ---------------------------------------------------------------------
strategy_rows = [
    r2_paired,
    paired_against(level_df("analysis_2_community_vax", "comm_vax_40"), ENH_OPS,
                   "Community vaccination\n40% coverage", "Enhanced operations"),
    paired_against(level_df("analysis_3_hybrid", "hybrid_40"), ENH_OPS,
                   "Hybrid\nring + 40% community", "Enhanced operations"),
]
df_strategy = pd.concat(strategy_rows, ignore_index=True)
strategy_order = [x["Strategy"].iloc[0] for x in strategy_rows]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.0]})

abs_rows = [
    df_r2_base.assign(Strategy="No vaccine\nmoderate operations"),
    df_r2_ring.assign(Strategy="Reactive ring (Radius 2)\nmoderate operations"),
    ENH_OPS.assign(Strategy="No vaccine\nenhanced operations"),
    level_df("analysis_2_community_vax", "comm_vax_40").assign(Strategy="Community vaccination\n40% coverage"),
]
df_abs = pd.concat(abs_rows, ignore_index=True)
abs_order = [x["Strategy"].iloc[0] for x in abs_rows]

sns.violinplot(
    data=df_abs, x="deaths_count", y="Strategy", hue="Strategy", order=abs_order, ax=axes[0],
    palette=["#5f6368", "#4f6d7a", "#5f6368", "#2a9d8f"],
    inner="quart", orient="h", density_norm="width", cut=0, legend=False
)
axes[0].set_xlabel("Deaths per 100,000-person network")
axes[0].set_ylabel("")
axes[0].set_title("A. Absolute mortality", loc="left")

sns.violinplot(
    data=df_strategy, x="Change in mortality (%)", y="Strategy", hue="Strategy", order=strategy_order, ax=axes[1],
    palette=["#4f6d7a", "#2a9d8f", "#e76f51"],
    inner="quart", orient="h", density_norm="width", cut=0, legend=False
)
axes[1].set_xlabel("Change in mortality (%)")
axes[1].set_ylabel("")
axes[1].set_title("B. Incremental benefit versus no-vaccine comparator", loc="left")
axes[1].set_xlim(-100, 150)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].axvline(0, color="black", linewidth=0.8)

fig.suptitle("Community vaccination changes outbreak mortality more than reactive ring vaccination")
fig.tight_layout()
save(fig, "fig2_strategies")


# ---------------------------------------------------------------------
# Figure 6: Timing
# ---------------------------------------------------------------------
timing_specs = [
    ("pre_emptive_day0", "Day 0\n(Pre-emptive)"),
    ("reactive_detect_delay_0", "At first\ndetection"),
    ("reactive_detect_delay_7", "Detection\n+7 days"),
    ("reactive_detect_delay_14", "Detection\n+14 days"),
]
df_time = pd.concat(
    [
        paired_against(level_df("analysis_4_timing", level), ENH_OPS, label, "Enhanced operations")
        for level, label in timing_specs
    ],
    ignore_index=True,
)
time_order = [label for _, label in timing_specs]
time_summary = df_time.groupby("Strategy", observed=False).agg(
    median=("Change in mortality (%)", "median"),
    p25=("Change in mortality (%)", lambda x: np.percentile(x.dropna(), 25)),
    p75=("Change in mortality (%)", lambda x: np.percentile(x.dropna(), 75))
).loc[time_order].reset_index()

# Extract Radius 2 Ring as reference line
r2_median = r2_paired["Change in mortality (%)"].median()

fig6, ax6 = plt.subplots(figsize=(6, 4.5))
x = np.arange(len(time_order))
med = time_summary["median"].to_numpy(dtype=float)
p25 = time_summary["p25"].to_numpy(dtype=float)
p75 = time_summary["p75"].to_numpy(dtype=float)

ax6.plot(x, med, marker="o", color="#9d174d", linewidth=2.5, label="Community vaccination (50%)")
ax6.fill_between(x, p25, p75, color="#9d174d", alpha=0.2)
ax6.axhline(r2_median, color="#4f6d7a", linestyle="--", linewidth=2, label="Reactive ring (Radius 2)")
ax6.axhline(0, color="black", linewidth=0.8)

ax6.set_xticks(x)
ax6.set_xticklabels(time_order)
ax6.set_title("Figure 6. Timing of community vaccination")
ax6.set_ylabel("Median change in mortality (%)")
ax6.legend(frameon=False)
fig6.tight_layout()
save(fig6, "fig6_timing")

# ---------------------------------------------------------------------
# Figure 7: Dose-impact frontier
# ---------------------------------------------------------------------
dose_comm = raw[raw["scenario"] == "analysis_5_dose_efficiency"].copy()
dose_comm["Strategy"] = np.where(dose_comm["level"].str.startswith("ring"), "Reactive ring", "Community vaccination")
dose_hybrid = raw[raw["scenario"] == "analysis_3_hybrid"].copy()
dose_hybrid["Strategy"] = "Hybrid"

dose = pd.concat([dose_comm, dose_hybrid], ignore_index=True)
dose["Coverage"] = dose["level"].str.extract(r"_(\d+)$").astype(int)

dose = pd.concat(
    [
        paired_against(group, ENH_OPS, name, "Enhanced operations")
        for name, group in dose.groupby("Strategy", observed=False)
    ],
    ignore_index=True,
)
dose["Coverage"] = dose["level"].str.extract(r"_(\d+)$").astype(int)

dose_summary = (
    dose.groupby(["Strategy", "Coverage"], observed=False)
    .agg(
        vaccines=("vaccines", "median"),
        pct_reduction=("Change in mortality (%)", "median"),
    )
    .reset_index()
)

fig7, ax7 = plt.subplots(figsize=(7, 5))
sns.lineplot(
    data=dose_summary,
    x="vaccines",
    y="pct_reduction",
    hue="Strategy",
    marker="o",
    linewidth=2.2,
    ax=ax7,
    palette={"Reactive ring": "#4f6d7a", "Community vaccination": "#2a9d8f", "Hybrid": "#e76f51"},
)

# Label points by coverage
for i, row in dose_summary.iterrows():
    ax7.text(row["vaccines"] + 150, row["pct_reduction"], f"{row['Coverage']}%", 
             fontsize=8, va="center", ha="left")

ax7.axhline(0, color="black", linewidth=0.8)
ax7.set_title("Figure 7. Dose-impact frontier")
ax7.set_xlabel("Median vaccine courses delivered")
ax7.set_ylabel("Median change in mortality (%)")
ax7.legend(frameon=False, loc="lower right")

fig7.tight_layout()
save(fig7, "fig7_dose_impact")
