import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from paths import result_path


BASE_SEED = 20260705
N_REPLICATES = 10000
N_WORKERS = 8
POPULATION_SIZE = 5000
BASE_CFR = 0.4539079029615015
VACCINE_EFFECT = 0.45
VAX_CFR = BASE_CFR * (1.0 - VACCINE_EFFECT)
BASELINE_TAU = 0.25
HOUSEHOLD_MEAN = 5.0
COMMUNITY_MEAN = 5.0
COMMUNITY_VARIANCE = 25.0

RAW_PATH = result_path("operational_ring_sensitivity_raw.csv")
SUMMARY_PATH = result_path("operational_ring_sensitivity_summary.csv")
FIG_DIR = Path("figures/polished")
FIG_PATH = FIG_DIR / "fig_operational_ring_polished.png"
PDF_PATH = FIG_DIR / "fig_operational_ring_polished.pdf"

COLORS = {
    "ring1": "#536F79",
    "ring2": "#C96549",
    "grid": "#E5E7EB",
    "text": "#111827",
}


def ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)


def load_rt_array():
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        return json.load(f)["tau_array"]


def scenario_definitions():
    scenarios = []
    detection_targets = [0.4, 0.6, 0.7, 0.8, 0.9]
    contact_targets = [0.4, 0.6, 0.8, 0.9]

    for target in detection_targets:
        scenarios.append(
            {
                "analysis": "index_case_detection",
                "level": f"detection_{target:.1f}",
                "radius": 0,
                "detection": target,
                "tracing": 0.8,
                "vaccine_effect": 0.0,
                "vax_cfr": BASE_CFR,
                "max_vaccines": 0,
            }
        )
        for radius in [1, 2]:
            scenarios.append(
                {
                    "analysis": "index_case_detection",
                    "level": f"detection_{target:.1f}",
                    "radius": radius,
                    "detection": target,
                    "tracing": 0.8,
                    "vaccine_effect": VACCINE_EFFECT,
                    "vax_cfr": VAX_CFR,
                    "max_vaccines": None,
                }
            )

    for target in contact_targets:
        scenarios.append(
            {
                "analysis": "contact_coverage",
                "level": f"coverage_{target:.1f}",
                "radius": 0,
                "detection": 0.7,
                "tracing": target,
                "vaccine_effect": 0.0,
                "vax_cfr": BASE_CFR,
                "max_vaccines": 0,
            }
        )
        for radius in [1, 2]:
            scenarios.append(
                {
                    "analysis": "contact_coverage",
                    "level": f"coverage_{target:.1f}",
                    "radius": radius,
                    "detection": 0.7,
                    "tracing": target,
                    "vaccine_effect": VACCINE_EFFECT,
                    "vax_cfr": VAX_CFR,
                    "max_vaccines": None,
                }
            )
    return scenarios


RT_ARRAY = load_rt_array()
GLOBAL_GRAPH = generate_network(
    POPULATION_SIZE,
    household_mean=HOUSEHOLD_MEAN,
    community_mean=COMMUNITY_MEAN,
    community_variance=COMMUNITY_VARIANCE,
)


def run_one(args):
    scenario, replicate = args
    seed = BASE_SEED + replicate
    np.random.seed(seed)
    cases, deaths, vaccines = simulate_ring_vaccination(
        GLOBAL_GRAPH,
        rt_array=RT_ARRAY,
        baseline_tau=BASELINE_TAU,
        incubation_period=8.5,
        infectious_period=6.0,
        uptake=1.0,
        vaccine_effect=scenario["vaccine_effect"],
        reporting_rate=ramp(scenario["detection"]),
        tracing_coverage=ramp(scenario["tracing"]),
        vaccine_acceptability=1.0,
        detection_delay=4.0,
        ring_radius=scenario["radius"],
        max_daily_traces=1000,
        max_vaccines=scenario["max_vaccines"],
        base_CFR=BASE_CFR,
        vax_CFR=scenario["vax_cfr"],
        initial_infected=5,
        initial_exposed=0,
        max_sim_time=90,
        engine="cpp",
        seed=seed,
        allow_pep=True,
        sigmoidal_k=0.5,
        sigmoidal_d0=10.0,
    )
    return {
        "analysis": scenario["analysis"],
        "level": scenario["level"],
        "radius": scenario["radius"],
        "replicate": replicate,
        "seed": seed,
        "population_size": POPULATION_SIZE,
        "detection": scenario["detection"],
        "contact_coverage": scenario["tracing"],
        "vaccine_effect": scenario["vaccine_effect"],
        "base_cfr": BASE_CFR,
        "vax_cfr": scenario["vax_cfr"],
        "cases_percent": cases * 100.0,
        "deaths_percent": deaths * 100.0,
        "vaccines": vaccines,
        "vaccines_per_100k": vaccines / POPULATION_SIZE * 100000.0,
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_raw():
    scenarios = scenario_definitions()
    tasks = [(scenario, replicate) for scenario in scenarios for replicate in range(N_REPLICATES)]
    rows = []
    with Pool(processes=N_WORKERS) as pool:
        for idx, row in enumerate(pool.imap_unordered(run_one, tasks, chunksize=50), start=1):
            rows.append(row)
            if idx % 20000 == 0:
                print(f"completed {idx}/{len(tasks)}")
    rows.sort(key=lambda r: (r["analysis"], r["level"], r["radius"], r["replicate"]))
    write_csv(RAW_PATH, rows)
    print(f"Wrote {RAW_PATH}")


def paired_reductions(raw):
    rows = []
    for (analysis, level), group in raw.groupby(["analysis", "level"], observed=False):
        baseline = group[group["radius"] == 0][["seed", "cases_percent", "deaths_percent"]].rename(
            columns={"cases_percent": "baseline_cases", "deaths_percent": "baseline_deaths"}
        )
        for radius in [1, 2]:
            sub = group[group["radius"] == radius].merge(baseline, on="seed", how="inner")
            sub["case_reduction"] = np.where(
                sub["baseline_cases"] > 0,
                (sub["baseline_cases"] - sub["cases_percent"]) / sub["baseline_cases"] * 100.0,
                np.nan,
            )
            sub["mortality_reduction"] = np.where(
                sub["baseline_deaths"] > 0,
                (sub["baseline_deaths"] - sub["deaths_percent"]) / sub["baseline_deaths"] * 100.0,
                np.nan,
            )
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def summarise(raw):
    paired = paired_reductions(raw)
    summary = (
        paired.groupby(["analysis", "level", "radius", "detection", "contact_coverage"], observed=False)
        .agg(
            n=("seed", "count"),
            case_reduction_median=("case_reduction", "median"),
            case_reduction_p25=("case_reduction", lambda x: np.nanpercentile(x, 25)),
            case_reduction_p75=("case_reduction", lambda x: np.nanpercentile(x, 75)),
            mortality_reduction_median=("mortality_reduction", "median"),
            mortality_reduction_p25=("mortality_reduction", lambda x: np.nanpercentile(x, 25)),
            mortality_reduction_p75=("mortality_reduction", lambda x: np.nanpercentile(x, 75)),
            vaccines_per_100k_median=("vaccines_per_100k", "median"),
            vaccines_per_100k_p25=("vaccines_per_100k", lambda x: np.nanpercentile(x, 25)),
            vaccines_per_100k_p75=("vaccines_per_100k", lambda x: np.nanpercentile(x, 75)),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote {SUMMARY_PATH}")
    return summary


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
    )


def plot_summary(summary):
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), gridspec_kw={"wspace": 0.36})
    labels = {1: "Ring 1", 2: "Ring 2"}
    colors = {1: COLORS["ring1"], 2: COLORS["ring2"]}

    det = summary[summary["analysis"] == "index_case_detection"].sort_values(["radius", "detection"])
    cov = summary[summary["analysis"] == "contact_coverage"].sort_values(["radius", "contact_coverage"])

    for radius in [1, 2]:
        sub = det[det["radius"] == radius]
        x = sub["detection"].to_numpy() * 100
        y = sub["mortality_reduction_median"].to_numpy()
        axes[0].plot(x, y, marker="o", linewidth=2.0, markersize=4.5, color=colors[radius], label=labels[radius])
        axes[0].fill_between(
            x,
            sub["mortality_reduction_p25"].to_numpy(),
            sub["mortality_reduction_p75"].to_numpy(),
            color=colors[radius],
            alpha=0.16,
            linewidth=0,
        )

    for radius in [1, 2]:
        sub = cov[cov["radius"] == radius]
        x = sub["contact_coverage"].to_numpy() * 100
        y = sub["mortality_reduction_median"].to_numpy()
        axes[1].plot(x, y, marker="o", linewidth=2.0, markersize=4.5, color=colors[radius], label=labels[radius])
        axes[1].fill_between(
            x,
            sub["mortality_reduction_p25"].to_numpy(),
            sub["mortality_reduction_p75"].to_numpy(),
            color=colors[radius],
            alpha=0.16,
            linewidth=0,
        )

    for radius in [1, 2]:
        sub = cov[cov["radius"] == radius]
        axes[2].plot(
            sub["vaccines_per_100k_median"],
            sub["mortality_reduction_median"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=colors[radius],
            label=labels[radius],
        )
        for _, row in sub.iterrows():
            if row["contact_coverage"] in [0.4, 0.9]:
                axes[2].text(
                    row["vaccines_per_100k_median"] + 1300,
                    row["mortality_reduction_median"],
                    f"{int(row['contact_coverage'] * 100)}%",
                    fontsize=7.2,
                    ha="left",
                    va="center",
                    color=colors[radius],
                )

    axes[0].set_xlabel("Index case detection (%)")
    axes[0].set_ylabel("Reduction in deaths (%)")
    axes[0].set_xlim(36, 94)
    axes[0].set_xticks([40, 60, 80])
    axes[0].set_ylim(-5, 100)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].set_xlabel("Contact coverage (%)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(36, 94)
    axes[1].set_xticks([40, 60, 80])
    axes[1].set_ylim(-5, 100)

    axes[2].set_xlabel("Vaccine courses per 100,000")
    axes[2].set_ylabel("")
    axes[2].set_ylim(-5, 100)

    for ax, label in zip(axes, ["A", "B", "C"]):
        panel_label(ax, label)
        ax.axhline(0, color="#4B5563", linewidth=0.7)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(FIG_PATH, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {FIG_PATH}")
    print(f"Wrote {PDF_PATH}")


def main():
    if RAW_PATH.exists():
        raw = pd.read_csv(RAW_PATH)
        print(f"Using existing {RAW_PATH}")
    else:
        generate_raw()
        raw = pd.read_csv(RAW_PATH)
    summary = summarise(raw)
    plot_summary(summary)


if __name__ == "__main__":
    main()
