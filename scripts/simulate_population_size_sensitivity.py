import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ebola_stochastic_ring import calibrate_tau, generate_network, simulate_ring_vaccination
from paths import DATA_DIR, result_path


BASE_SEED = 20260629
N_TRIALS = int(os.environ.get("POP_SIZE_SENSITIVITY_REPS", "1000"))
N_WORKERS = int(os.environ.get("POP_SIZE_SENSITIVITY_WORKERS", "8"))
POPULATION_SIZES = [2000, 5000, 10000]
RADII = [1, 2]
TRANSMISSION_MODE = os.environ.get("POP_SIZE_TRANSMISSION", "rt").lower()
HOUSEHOLD_MEAN = float(os.environ.get("NETWORK_HOUSEHOLD_MEAN", "5.0"))
COMMUNITY_MEAN = float(os.environ.get("NETWORK_COMMUNITY_MEAN", "5.0"))
COMMUNITY_VARIANCE = float(os.environ.get("NETWORK_COMMUNITY_VARIANCE", "25.0"))
TAU_CALIBRATION_TRIALS = int(os.environ.get("POP_SIZE_TAU_CALIBRATION_TRIALS", "50"))


def load_cfr_parameters():
    try:
        with open(result_path("fitted_parameters.json"), "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        params = {}
    base_cfr = float(params.get("base_CFR", params.get("latest_adjusted_cfr", 0.454)))
    return base_cfr, base_cfr * 0.5


BASE_CFR, VAX_CFR = load_cfr_parameters()


def load_rt_array():
    if TRANSMISSION_MODE != "rt":
        return None
    with open(result_path("fitted_parameters.json"), "r") as f:
        params = json.load(f)
    return params.get("Rt_array")


RT_ARRAY = load_rt_array()
RT_MAX = max(RT_ARRAY) if RT_ARRAY else 1.66


def calibrate_tau_for_population(population_size):
    if TRANSMISSION_MODE != "rt":
        return float(os.environ.get("POP_SIZE_BASELINE_TAU", "0.08"))
    graph = generate_network(
        population_size,
        household_mean=HOUSEHOLD_MEAN,
        community_mean=COMMUNITY_MEAN,
        community_variance=COMMUNITY_VARIANCE,
    )
    return calibrate_tau(graph, RT_MAX, 1.0 / 6.0, num_trials=TAU_CALIBRATION_TRIALS)


BASELINE_TAU_BY_N = {n: calibrate_tau_for_population(n) for n in POPULATION_SIZES}


def run_sim(args):
    population_size, radius, seed = args
    np.random.seed(seed)
    graph = generate_network(
        population_size,
        household_mean=HOUSEHOLD_MEAN,
        community_mean=COMMUNITY_MEAN,
        community_variance=COMMUNITY_VARIANCE,
    )
    cases, deaths, vaccines = simulate_ring_vaccination(
        graph,
        rt_array=RT_ARRAY,
        baseline_tau=BASELINE_TAU_BY_N[population_size],
        incubation_period=8.5,
        infectious_period=6.0,
        uptake=0.8,
        efficacy=0.4,
        reporting_rate=0.7,
        detection_delay=4.0,
        ring_radius=radius,
        max_daily_traces=1000,
        max_vaccines=None,
        base_CFR=BASE_CFR,
        vax_CFR=VAX_CFR,
    )
    return {
        "population_size": population_size,
        "radius": radius,
        "seed": seed,
        "transmission_mode": TRANSMISSION_MODE,
        "baseline_tau": BASELINE_TAU_BY_N[population_size],
        "rt_max": RT_MAX,
        "household_mean": HOUSEHOLD_MEAN,
        "community_mean": COMMUNITY_MEAN,
        "community_variance": COMMUNITY_VARIANCE,
        "cases_percent": cases * 100.0,
        "deaths_percent": deaths * 100.0,
        "vaccines": vaccines,
        "vaccines_percent": vaccines / population_size * 100.0,
    }


def summarize(rows):
    summary = []
    for population_size in POPULATION_SIZES:
        for radius in RADII:
            subset = [
                row for row in rows
                if row["population_size"] == population_size and row["radius"] == radius
            ]
            for metric in ["cases_percent", "deaths_percent", "vaccines_percent"]:
                values = np.array([row[metric] for row in subset], dtype=float)
                summary.append({
                    "population_size": population_size,
                    "radius": radius,
                    "metric": metric,
                    "n": len(values),
                    "transmission_mode": subset[0]["transmission_mode"],
                    "baseline_tau": subset[0]["baseline_tau"],
                    "rt_max": subset[0]["rt_max"],
                    "household_mean": subset[0]["household_mean"],
                    "community_mean": subset[0]["community_mean"],
                    "community_variance": subset[0]["community_variance"],
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p2_5": float(np.percentile(values, 2.5)),
                    "p97_5": float(np.percentile(values, 97.5)),
                })
    return summary


def write_outputs(rows, summary):
    raw_csv = result_path("population_size_sensitivity_raw.csv")
    summary_csv = result_path("population_size_sensitivity_summary.csv")
    summary_md = result_path("population_size_sensitivity_summary.md")

    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    with open(summary_md, "w") as f:
        f.write("# Population Size Sensitivity Analysis\n\n")
        f.write(
            "Focused robustness analysis for the primary 40% vaccine-efficacy, "
            "70% case-detection scenario. Values are percentages of the simulated "
            "local transmission network. Intervals are empirical 2.5th and 97.5th "
            f"percentiles across {N_TRIALS} stochastic replicates per population-size/radius combination. "
            f"Transmission mode: {TRANSMISSION_MODE}; peak Rt target: {RT_MAX:.2f}; "
            f"community degree mean {COMMUNITY_MEAN:.1f}, variance {COMMUNITY_VARIANCE:.1f}.\n\n"
        )
        f.write("| Population size | Radius | Metric | n | Mean (2.5-97.5%) |\n")
        f.write("|---:|---:|---|---:|---:|\n")
        for row in summary:
            f.write(
                "| {population_size} | {radius} | {metric} | {n} | "
                "{mean:.2f} ({p2_5:.2f}-{p97_5:.2f}) |\n".format(**row)
            )

    np.savez(
        DATA_DIR / "population_size_sensitivity.npz",
        population_size=np.array([row["population_size"] for row in rows]),
        radius=np.array([row["radius"] for row in rows]),
        seed=np.array([row["seed"] for row in rows]),
        cases_percent=np.array([row["cases_percent"] for row in rows]),
        deaths_percent=np.array([row["deaths_percent"] for row in rows]),
        vaccines=np.array([row["vaccines"] for row in rows]),
        vaccines_percent=np.array([row["vaccines_percent"] for row in rows]),
        baseline_tau=np.array([row["baseline_tau"] for row in rows]),
        rt_max=np.array([row["rt_max"] for row in rows]),
    )
    return raw_csv, summary_csv, summary_md


def main():
    args = []
    seed = BASE_SEED
    for population_size in POPULATION_SIZES:
        for radius in RADII:
            for _ in range(N_TRIALS):
                seed += 1
                args.append((population_size, radius, seed))

    if N_WORKERS <= 1:
        rows = [run_sim(arg) for arg in args]
    else:
        rows = []
        with Pool(processes=N_WORKERS) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_sim, args, chunksize=10), start=1):
                rows.append(row)
                if idx % 500 == 0:
                    print(f"Completed {idx}/{len(args)} simulations", flush=True)

    summary = summarize(rows)
    paths = write_outputs(rows, summary)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
