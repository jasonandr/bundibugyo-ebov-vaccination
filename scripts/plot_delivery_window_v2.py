import os
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ebola_stochastic_ring as sim


OUT_DIR = "figures/new_analyses"
CACHE_PATH = "data_and_results/fig5_delivery_window_v2_raw.csv"
FIG_PATH = os.path.join(OUT_DIR, "fig5_delivery_window_v2.png")
PDF_PATH = os.path.join(OUT_DIR, "fig5_delivery_window_v2.pdf")

N = int(os.environ.get("FIG5_WINDOW_N", "10000"))
REPS = int(os.environ.get("FIG5_WINDOW_REPS", "250"))
WORKERS = int(os.environ.get("FIG5_WINDOW_WORKERS", str(max(1, mp.cpu_count() - 1))))
BASE_SEED = int(os.environ.get("FIG5_WINDOW_SEED", "20260704"))

COLORS = {
    "Reactive ring": "#4F6D7A",
    "Community vaccination": "#1F9D8A",
    "Before exposure": "#1F9D8A",
    "Incubation": "#F59E0B",
    "After symptoms": "#D96B4A",
    "After outcome": "#D1D5DB",
    "Text": "#111827",
    "Muted": "#6B7280",
    "Grid": "#D1D5DB",
}


def enhanced_ops():
    reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7] * 76
    tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8] * 76
    tau_array = np.linspace(0.12, 0.05, 91).tolist()
    return reporting, tracing, tau_array


def run_single(rep, strategy):
    seed = BASE_SEED + rep
    np.random.seed(seed)
    graph = sim.generate_network(N)
    reporting, tracing, tau_array = enhanced_ops()

    common = dict(
        initial_infected=5,
        rt_array=tau_array,
        baseline_tau=0.25,
        vaccine_effect=0.0,
        base_CFR=0.454,
        reporting_rate=reporting,
        tracing_coverage=tracing,
        max_sim_time=90,
        seed=seed,
        engine="cpp",
        return_mechanism=True,
        sigmoidal_k=0.5,
        sigmoidal_d0=10.0,
    )

    if strategy == "Reactive ring":
        res = sim.simulate_ring_vaccination(
            graph,
            ring_radius=2,
            community_vax_coverage=0.0,
            **common,
        )
    else:
        res = sim.simulate_ring_vaccination(
            graph,
            ring_radius=1,
            max_vaccines=0,
            community_vax_coverage=0.40,
            community_vax_trigger=1,
            community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            **common,
        )

    exposure = np.asarray(res["exposure_time"], dtype=float)
    onset = np.asarray(res["onset_time"], dtype=float)
    outcome = np.asarray(res["recovery_or_death_time"], dtype=float)
    died = np.asarray(res["died"], dtype=bool)
    vaccination = np.asarray(res["vaccination_time"], dtype=float)

    cases = (exposure >= 0) & (onset >= 0)
    vaccinated = cases & (vaccination >= 0)
    fatal_cases = cases & died & (outcome >= 0)

    n_cases = int(cases.sum())
    n_vaccinated = int(vaccinated.sum())
    n_fatal = int(fatal_cases.sum())

    def pct(num, den):
        return np.nan if den == 0 else 100.0 * float(num) / float(den)

    phase_counts = {
        "Before exposure": int((vaccinated & (vaccination < exposure)).sum()),
        "Incubation": int((vaccinated & (vaccination >= exposure) & (vaccination < onset)).sum()),
        "After symptoms": int((vaccinated & (vaccination >= onset) & ((outcome < 0) | (vaccination < outcome))).sum()),
        "After outcome": int((vaccinated & (outcome >= 0) & (vaccination >= outcome)).sum()),
    }
    phase_rows = [
        {
            "record_type": "phase",
            "replicate": rep,
            "strategy": strategy,
            "phase": phase,
            "percent": pct(count, n_vaccinated),
            "denominator": n_vaccinated,
        }
        for phase, count in phase_counts.items()
    ]

    milestone_rows = [
        {
            "record_type": "milestone",
            "replicate": rep,
            "strategy": strategy,
            "milestone": "Before exposure",
            "percent": pct((vaccinated & (vaccination < exposure)).sum(), n_cases),
            "denominator": n_cases,
        },
        {
            "record_type": "milestone",
            "replicate": rep,
            "strategy": strategy,
            "milestone": "Before symptoms",
            "percent": pct((vaccinated & (vaccination < onset)).sum(), n_cases),
            "denominator": n_cases,
        },
        {
            "record_type": "milestone",
            "replicate": rep,
            "strategy": strategy,
            "milestone": "Before death",
            "percent": pct((fatal_cases & (vaccination >= 0) & (vaccination < outcome)).sum(), n_fatal),
            "denominator": n_fatal,
        },
    ]

    timing_rows = []
    idx = np.where(vaccinated)[0]
    if len(idx) > 250:
        idx = np.random.choice(idx, size=250, replace=False)
    for node in idx:
        timing_rows.append(
            {
                "record_type": "timing",
                "replicate": rep,
                "strategy": strategy,
                "days_relative_to_exposure": vaccination[node] - exposure[node],
                "days_relative_to_onset": vaccination[node] - onset[node],
            }
        )

    return phase_rows + milestone_rows + timing_rows


def build_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tasks = [(rep, strategy) for strategy in ["Reactive ring", "Community vaccination"] for rep in range(REPS)]
    rows = []
    with mp.Pool(WORKERS) as pool:
        for result in pool.starmap(run_single, tasks, chunksize=5):
            rows.extend(result)
    data = pd.DataFrame(rows)
    data.to_csv(CACHE_PATH, index=False)
    return data


def load_data():
    if os.path.exists(CACHE_PATH) and os.environ.get("FIG5_WINDOW_FORCE_RERUN", "0") != "1":
        return pd.read_csv(CACHE_PATH, low_memory=False)
    return build_cache()


def panel_label(ax, label):
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", color=COLORS["Text"])


def draw_timeline(ax):
    panel_label(ax, "A")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def arrow(x0, x1, y, color, lw=1.8):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13, lw=lw, color=color))

    ax.text(3, 8.6, "Index case", ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax.text(3, 4.4, "Secondary contact", ha="left", va="center", fontsize=9.5, fontweight="bold")
    arrow(18, 93, 8.0, COLORS["Text"])
    arrow(18, 93, 3.8, COLORS["Text"])

    index_events = [(24, "Exposure"), (49, "Symptoms"), (64, "Detection"), (78, "Tracing"), (88, "Ring\nvaccination")]
    secondary_events = [(39, "Exposure"), (62, "Symptoms"), (84, "Death/\nrecovery")]

    for x, label in index_events:
        ax.plot([x, x], [7.65, 8.35], color=COLORS["Text"], lw=1)
        ax.text(x, 8.55, label, ha="center", va="bottom", fontsize=8.2)

    for x, label in secondary_events:
        ax.plot([x, x], [3.45, 4.15], color=COLORS["Text"], lw=1)
        ax.text(x, 3.2, label, ha="center", va="top", fontsize=8.2)

    ax.add_patch(Rectangle((18, 4.95), 21, 0.6, facecolor="#DFF3EF", edgecolor="none"))
    ax.add_patch(Rectangle((39, 4.95), 23, 0.6, facecolor="#FEF3C7", edgecolor="none"))
    ax.add_patch(Rectangle((62, 4.95), 22, 0.6, facecolor="#FEE2E2", edgecolor="none"))
    ax.text(28.5, 5.95, "before exposure", ha="center", va="bottom", fontsize=8, color=COLORS["Before exposure"], fontweight="bold")
    ax.text(50.5, 5.95, "incubation", ha="center", va="bottom", fontsize=8, color="#92400E", fontweight="bold")
    ax.text(73, 5.95, "after symptoms", ha="center", va="bottom", fontsize=8, color="#9F4B34", fontweight="bold")

    arrow(19, 35, 6.8, COLORS["Community vaccination"], lw=2.8)
    ax.text(27, 7.1, "community vaccination", ha="center", va="bottom", fontsize=8.5, color=COLORS["Community vaccination"], fontweight="bold")
    arrow(64, 88, 6.8, COLORS["Reactive ring"], lw=2.8)
    ax.text(76, 7.1, "reactive ring", ha="center", va="bottom", fontsize=8.5, color=COLORS["Reactive ring"], fontweight="bold")


def draw_phase(ax, phase):
    panel_label(ax, "B")
    phase_order = ["Before exposure", "Incubation", "After symptoms", "After outcome"]
    strategy_order = ["Community vaccination", "Reactive ring"]
    summary = (
        phase.groupby(["strategy", "phase"], observed=False)["percent"]
        .agg(median="median")
        .reset_index()
    )
    pivot = summary.pivot(index="strategy", columns="phase", values="median").reindex(strategy_order)[phase_order].fillna(0)

    y = np.arange(len(strategy_order))
    left = np.zeros(len(strategy_order))
    for ph in phase_order:
        vals = pivot[ph].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=COLORS[ph], edgecolor="white", linewidth=1.2, height=0.56, label=ph)
        for i, val in enumerate(vals):
            if val >= 8:
                x = left[i] + val / 2
                color = "white" if ph in ["Before exposure", "After symptoms"] else COLORS["Text"]
                ax.text(x, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(strategy_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Timing phase among vaccinated secondary cases (%)")
    ax.grid(axis="x", color=COLORS["Grid"], alpha=0.45, lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.0,
    )


def draw_milestones(ax, milestone):
    panel_label(ax, "C")
    milestone_order = ["Before exposure", "Before symptoms", "Before death"]
    strategy_order = ["Community vaccination", "Reactive ring"]
    offsets = {"Community vaccination": -0.12, "Reactive ring": 0.12}
    x_lookup = {m: i for i, m in enumerate(milestone_order)}
    summary = (
        milestone.groupby(["strategy", "milestone"], observed=False)["percent"]
        .agg(median="median", p25=lambda x: np.nanpercentile(x, 25), p75=lambda x: np.nanpercentile(x, 75))
        .reset_index()
    )

    for strategy in strategy_order:
        sub = summary[summary["strategy"] == strategy].set_index("milestone").reindex(milestone_order)
        xs = np.array([x_lookup[m] + offsets[strategy] for m in milestone_order])
        med = sub["median"].to_numpy(dtype=float)
        p25 = sub["p25"].to_numpy(dtype=float)
        p75 = sub["p75"].to_numpy(dtype=float)
        color = COLORS[strategy]
        ax.errorbar(xs, med, yerr=[med - p25, p75 - med], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, ms=6, label=strategy)
        for x, m in zip(xs, med):
            ax.text(x, m + 3, f"{m:.0f}%", ha="center", va="bottom", fontsize=8.2, color=color, fontweight="bold")

    ax.set_xticks(range(len(milestone_order)))
    ax.set_xticklabels(["Before\nexposure", "Before\nsymptoms", "Before\ndeath"])
    ax.set_ylim(0, 55)
    ax.set_ylabel("Vaccinated before milestone (%)")
    ax.grid(axis="y", color=COLORS["Grid"], alpha=0.45, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        2,
        -8,
        "Before-death denominator: fatal secondary cases",
        ha="center",
        va="top",
        fontsize=7.2,
        color=COLORS["Muted"],
        transform=ax.transData,
    )


def plot():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    data = load_data()
    phase = data[data["record_type"] == "phase"].copy()
    milestone = data[data["record_type"] == "milestone"].copy()

    fig = plt.figure(figsize=(9.2, 5.8), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], width_ratios=[1.04, 0.96], hspace=0.40, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_timeline(ax_a)
    draw_phase(ax_b, phase)
    draw_milestones(ax_c, milestone)

    fig.text(
        0.01,
        0.01,
        "Diagnostic delivery-window analysis with vaccine protection disabled; initial infections without observed exposure times excluded.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=COLORS["Muted"],
    )
    fig.savefig(FIG_PATH, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"FIG5={FIG_PATH}")
    print(f"FIG5_PDF={PDF_PATH}")


if __name__ == "__main__":
    plot()
