import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from ebola_stochastic_ring import calibrate_tau, generate_network, simulate_ring_vaccination
from paths import result_path


BASE_SEED = 20260701
POPULATION_SIZE = int(os.environ.get("NETWORK_SENSITIVITY_N", "5000"))
N_REPLICATES = int(os.environ.get("NETWORK_SENSITIVITY_REPS", "1000"))
N_WORKERS = int(os.environ.get("NETWORK_SENSITIVITY_WORKERS", "8"))
HOUSEHOLD_MEAN = float(os.environ.get("NETWORK_HOUSEHOLD_MEAN", "5.0"))
COMMUNITY_GRID = [
    (3.0, 10.0),
    (3.0, 25.0),
    (5.0, 25.0),
    (8.0, 40.0),
    (8.0, 80.0),
]
RADII = [1, 2]
TAU_CALIBRATION_TRIALS = int(os.environ.get("NETWORK_SENSITIVITY_TAU_CALIBRATION_TRIALS", "75"))


def load_parameters():
    with open(result_path("fitted_parameters.json"), "r") as f:
        params = json.load(f)
    base_cfr = float(params.get("base_CFR", params.get("latest_adjusted_cfr", 0.454)))
    return params.get("Rt_array"), base_cfr, base_cfr * 0.5


RT_ARRAY, BASE_CFR, VAX_CFR = load_parameters()
RT_MAX = max(RT_ARRAY)


def calibrate_tau_for_network(community_mean, community_variance):
    np.random.seed(BASE_SEED + int(community_mean * 100) + int(community_variance))
    graph = generate_network(
        POPULATION_SIZE,
        household_mean=HOUSEHOLD_MEAN,
        community_mean=community_mean,
        community_variance=community_variance,
    )
    return calibrate_tau(graph, RT_MAX, 1.0 / 6.0, num_trials=TAU_CALIBRATION_TRIALS)


BASELINE_TAU = {
    (community_mean, community_variance): calibrate_tau_for_network(community_mean, community_variance)
    for community_mean, community_variance in COMMUNITY_GRID
}


def run_one(args):
    community_mean, community_variance, radius, vaccination, replicate, seed = args
    np.random.seed(seed)
    graph = generate_network(
        POPULATION_SIZE,
        household_mean=HOUSEHOLD_MEAN,
        community_mean=community_mean,
        community_variance=community_variance,
    )
    cases, deaths, vaccines = simulate_ring_vaccination(
        graph,
        rt_array=RT_ARRAY,
        baseline_tau=BASELINE_TAU[(community_mean, community_variance)],
        incubation_period=8.5,
        infectious_period=6.0,
        uptake=0.8 if vaccination else 0.0,
        efficacy=0.4 if vaccination else 0.0,
        reporting_rate=0.7,
        detection_delay=4.0,
        ring_radius=radius,
        max_daily_traces=1000,
        max_vaccines=None if vaccination else 0,
        base_CFR=BASE_CFR,
        vax_CFR=VAX_CFR,
    )
    return {
        "community_mean": community_mean,
        "community_variance": community_variance,
        "radius": radius,
        "vaccination": int(vaccination),
        "replicate": replicate,
        "seed": seed,
        "population_size": POPULATION_SIZE,
        "household_mean": HOUSEHOLD_MEAN,
        "baseline_tau": BASELINE_TAU[(community_mean, community_variance)],
        "rt_max": RT_MAX,
        "cases_percent": cases * 100.0,
        "deaths_percent": deaths * 100.0,
        "vaccines": vaccines,
        "vaccines_percent": vaccines / POPULATION_SIZE * 100.0,
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
    }


def summarize_rows(rows):
    baseline = {}
    for community_mean, community_variance in COMMUNITY_GRID:
        for radius in RADII:
            subset = [
                row for row in rows
                if row["community_mean"] == community_mean
                and row["community_variance"] == community_variance
                and row["radius"] == radius
                and row["vaccination"] == 0
            ]
            baseline[(community_mean, community_variance, radius)] = {
                "cases": np.mean([row["cases_percent"] for row in subset]),
                "deaths": np.mean([row["deaths_percent"] for row in subset]),
            }

    out = []
    for community_mean, community_variance in COMMUNITY_GRID:
        for radius in RADII:
            for vaccination in [0, 1]:
                subset = [
                    row for row in rows
                    if row["community_mean"] == community_mean
                    and row["community_variance"] == community_variance
                    and row["radius"] == radius
                    and row["vaccination"] == vaccination
                ]
                cases = np.array([row["cases_percent"] for row in subset], dtype=float)
                deaths = np.array([row["deaths_percent"] for row in subset], dtype=float)
                vaccines = np.array([row["vaccines_percent"] for row in subset], dtype=float)
                base = baseline[(community_mean, community_variance, radius)]
                if vaccination:
                    cases_metric = (base["cases"] - cases) / base["cases"] * 100.0
                    deaths_metric = (base["deaths"] - deaths) / base["deaths"] * 100.0
                else:
                    cases_metric = cases
                    deaths_metric = deaths
                cases_summary = summarize(cases_metric)
                deaths_summary = summarize(deaths_metric)
                vaccine_summary = summarize(vaccines)
                out.append({
                    "community_mean": community_mean,
                    "community_variance": community_variance,
                    "radius": radius,
                    "vaccination": vaccination,
                    "n": len(subset),
                    "population_size": POPULATION_SIZE,
                    "baseline_tau": BASELINE_TAU[(community_mean, community_variance)],
                    "rt_max": RT_MAX,
                    "cases_metric": "cases_averted_percent" if vaccination else "cases_percent",
                    "cases_mean": cases_summary["mean"],
                    "cases_p2_5": cases_summary["p2_5"],
                    "cases_p97_5": cases_summary["p97_5"],
                    "deaths_metric": "deaths_averted_percent" if vaccination else "deaths_percent",
                    "deaths_mean": deaths_summary["mean"],
                    "deaths_p2_5": deaths_summary["p2_5"],
                    "deaths_p97_5": deaths_summary["p97_5"],
                    "vaccines_percent_mean": vaccine_summary["mean"],
                    "vaccines_percent_p2_5": vaccine_summary["p2_5"],
                    "vaccines_percent_p97_5": vaccine_summary["p97_5"],
                })
    return out


def write_outputs(rows, summary_rows):
    raw_path = result_path("network_parameter_sensitivity_raw.csv")
    summary_path = result_path("network_parameter_sensitivity_summary.csv")
    summary_md_path = result_path("network_parameter_sensitivity_summary.md")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(summary_md_path, "w") as f:
        f.write("# Network Parameter Sensitivity\n\n")
        f.write(
            f"Rt-driven analyses with peak Rt target {RT_MAX:.2f}, N={POPULATION_SIZE}, "
            f"{N_REPLICATES} replicates per network/radius/vaccination setting, "
            "40% vaccine efficacy, 70% case detection, and 80% Radius 1 reach/acceptance.\n\n"
        )
        f.write("| Community mean | Community variance | Radius | Vaccination | n | Cases metric, mean (95% UI) | Deaths metric, mean (95% UI) | Vaccinated, mean % | Tau |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {community_mean:.1f} | {community_variance:.1f} | {radius} | {vaccination} | {n} | "
                "{cases_mean:.1f} ({cases_p2_5:.1f}-{cases_p97_5:.1f}) | "
                "{deaths_mean:.1f} ({deaths_p2_5:.1f}-{deaths_p97_5:.1f}) | "
                "{vaccines_percent_mean:.1f} | {baseline_tau:.4f} |\n".format(**row)
            )
    return raw_path, summary_path, summary_md_path


def main():
    args = []
    seed = BASE_SEED
    for community_mean, community_variance in COMMUNITY_GRID:
        for radius in RADII:
            for vaccination in [0, 1]:
                for replicate in range(N_REPLICATES):
                    seed += 1
                    args.append((community_mean, community_variance, radius, vaccination, replicate, seed))

    if N_WORKERS <= 1:
        rows = [run_one(arg) for arg in args]
    else:
        rows = []
        with Pool(processes=N_WORKERS) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_one, args, chunksize=10), start=1):
                rows.append(row)
                if idx % 500 == 0:
                    print(f"Completed {idx}/{len(args)} simulations", flush=True)

    summary_rows = summarize_rows(rows)
    for path in write_outputs(rows, summary_rows):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
