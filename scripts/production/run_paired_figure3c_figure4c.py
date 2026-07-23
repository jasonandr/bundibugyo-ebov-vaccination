"""Run same-seed comparator grids for Figure 3C and Figure 4C.

Each intervention replicate is paired with a base-operations, no-vaccination
replicate that uses the identical network and random seed.  This makes the
no-effect Figure 3C cell (risk multiplier 1, VE 0%) exactly zero by design
and removes stochastic comparator noise from the immune-onset sensitivity.
"""
import argparse
import csv
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ebola_stochastic_ring import simulate_ring_vaccination
from network_cache import load_cached_network

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
N, HORIZON = 100_000, 90
GRAPH = PARAMS = RT = None


def init(params, rt, cache):
    global GRAPH, PARAMS, RT
    GRAPH, PARAMS, RT = load_cached_network(cache), params, rt


def simulate(spec, seed):
    if spec.get("enhanced", False):
        reporting = np.linspace(.30, .70, 15).tolist() + [.70] * 76
        tracing = np.linspace(.30, .80, 15).tolist() + [.80] * 76
        detection_delay, tracing_delay = 2.0, 1.0
    else:
        reporting, tracing = [.30] * 91, .30
        detection_delay, tracing_delay = 4.0, 2.0
    return simulate_ring_vaccination(
        GRAPH, rt_array=RT, incubation_period=8.5, infectious_period=6.0,
        ring_radius=spec.get("radius", 2), efficacy=1.0,
        infection_efficacy_multiplier=spec.get("ve_i", 0.45),
        vax_CFR=float(PARAMS["base_CFR"]) * (1 - spec.get("ve_m", 0.45)),
        uptake=.8, reporting_rate=reporting, tracing_coverage=tracing,
        vaccine_acceptability=1.0, detection_delay=detection_delay, tracing_delay=tracing_delay,
        max_daily_traces=100, max_vaccines=spec.get("max_vaccines", 0),
        base_CFR=float(PARAMS["base_CFR"]), initial_infected=15,
        initial_exposed=15, max_sim_time=HORIZON, seed=seed, engine="cpp",
        sigmoidal_k=.5, sigmoidal_d0=spec.get("immune_midpoint", 10.0),
        allow_pep=True, community_vax_coverage=spec.get("coverage", 0.0),
        community_vax_trigger=1 if spec.get("coverage", 0.0) else 0,
        community_vax_delay=spec.get("delay", -1.0),
        community_vax_rollout_days=14.0 if spec.get("coverage", 0.0) else 0.0,
        risk_compensation_multiplier=spec.get("risk", 1.0),
        incubation_shape=2.0, infectious_shape=2.0,
    )


def task_run(task):
    analysis, spec, reps, seed0 = task
    base_spec = {"max_vaccines": 0}
    rows = []
    for replicate in range(reps):
        seed = seed0 + replicate
        base = simulate(base_spec, seed)[1] * N
        scenario = simulate(spec, seed)[1] * N
        rows.append({
            **spec, "analysis": analysis, "replicate": replicate, "seed": seed,
            "base_deaths": base, "scenario_deaths": scenario,
            "mortality_reduction_pct": 100 * (base - scenario) / base,
        })
    return analysis, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--network-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--midpoints", type=float, nargs="+", default=[5, 10, 14],
                        help="Immune-onset midpoints (days) for the Figure 4C sensitivity.")
    parser.add_argument("--fig4-only", action="store_true", help="Run only the common-seed Figure 4C sensitivity.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    params = json.loads((REPO / "data_and_results/fitted_parameters.json").read_text())
    rt = list(params["Rt_array"]) + [params["Rt_array"][-1]] * 25
    tasks = []
    for risk in np.linspace(1.0, 2.5, 7):
        for ve in np.linspace(0.0, .9, 7):
            tasks.append(("fig3_risk_compensation", {
                "coverage": .4, "ve_i": float(ve), "ve_m": float(ve),
                "risk": float(risk), "radius": 2, "max_vaccines": 0,
            }, args.replicates, 2026700000 + len(tasks) * 1000))
    for midpoint in args.midpoints:
        tasks.append(("fig4_immune_onset", {
            "radius": 2, "max_vaccines": None, "immune_midpoint": float(midpoint),
            "ve_i": .45, "ve_m": .45,
        }, args.replicates, 2026800000))
    if args.fig4_only:
        tasks = [task for task in tasks if task[0] == "fig4_immune_onset"]

    args.output_dir.mkdir(parents=True)
    grouped = {}
    with mp.Pool(args.workers, initializer=init, initargs=(params, rt, str(args.network_cache))) as pool:
        for analysis, rows in pool.imap_unordered(task_run, tasks):
            grouped.setdefault(analysis, []).extend(rows)
    for analysis, rows in grouped.items():
        with (args.output_dir / f"{analysis}_paired_raw.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicates_per_cell": args.replicates, "workers": args.workers,
        "network_cache": str(args.network_cache),
        "network": {"N": N, "household_mean": 5.2, "community_mean": 30.0, "community_variance": 160.0},
        "initial_state": {"infectious": 15, "exposed": 15},
        "rt_source": "fitted_parameters.json Rt_array; updated EpiNow2 median",
        "comparison": "same network and random seed paired to base operations with no vaccination; Figure 4C intervention is Ring 2 vaccination under base operations",
        "risk_rule": "vaccinated eligible contacts receive the requested risk weight in cohort demand and weighted destination allocation",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
