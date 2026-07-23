"""Supplementary S4: independent vaccine-efficacy grid (ve_i x ve_m), 13x13.

Community vaccination at 40% coverage under base operations.  Infection
efficacy is passed via `efficacy` (infection_efficacy_multiplier held at 1.0);
mortality efficacy is passed via `vax_CFR = base_CFR * (1 - ve_m)`, matching
the C++ path `cfr = base_CFR - tb * (base_CFR - vax_CFR)`.  Seeds are paired
across all cells and the no-vaccination base arm.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ebola_stochastic_ring import simulate_ring_vaccination

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
HORIZON = 90
SEED_BASE = 2026100000
VE_VALUES = [round(0.075 * i, 3) for i in range(13)]  # 0.0 .. 0.90
FIELDS = ["cell", "ve_i", "ve_m", "replicate_id", "simulation_seed",
          "cases", "deaths", "doses"]
WORKER_GRAPH = None
WORKER_PARAMS = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialise_worker(params, network_cache_path):
    global WORKER_GRAPH, WORKER_PARAMS
    from network_cache import load_cached_network
    WORKER_GRAPH = load_cached_network(network_cache_path)
    WORKER_PARAMS = params


def run_cell(task):
    ve_i, ve_m, replicates = task
    rt = list(WORKER_PARAMS["Rt_array"])
    rt.extend([rt[-1]] * max(0, HORIZON + 1 - len(rt)))
    base_cfr = float(WORKER_PARAMS["base_CFR"])
    base_arm = ve_i is None  # base operations, no vaccination
    community = not base_arm
    rows = []
    for replicate_id in range(replicates):
        sim_seed = SEED_BASE + replicate_id
        cases, deaths, doses, _ = simulate_ring_vaccination(
            WORKER_GRAPH, rt_array=rt, baseline_tau=0.25,
            incubation_period=8.5, infectious_period=6.0,
            ring_radius=2, efficacy=0.0 if base_arm else float(ve_i),
            vax_CFR=base_cfr if base_arm else base_cfr * (1.0 - float(ve_m)),
            uptake=0.8,
            reporting_rate=[0.30] * (HORIZON + 1),
            tracing_coverage=[0.30] * (HORIZON + 1),
            vaccine_acceptability=1.0, detection_delay=4.0, tracing_delay=2.0,
            max_daily_traces=100, max_vaccines=0,
            base_CFR=base_cfr,
            initial_infected=15, initial_exposed=15, max_sim_time=HORIZON,
            seed=sim_seed, engine="cpp",
            community_vax_coverage=0.4 if community else 0.0,
            community_vax_trigger=1 if community else 0,
            community_vax_delay=0.0 if community else -1.0,
            community_vax_rollout_days=14.0 if community else 0.0,
            sigmoidal_k=0.5, sigmoidal_d0=10.0, allow_pep=True,
            incubation_shape=2.0, infectious_shape=2.0,
        )
        rows.append({
            "cell": "base_no_vax" if base_arm else f"vei_{ve_i:.3f}_vem_{ve_m:.3f}",
            "ve_i": "" if base_arm else f"{ve_i:.3f}",
            "ve_m": "" if base_arm else f"{ve_m:.3f}",
            "replicate_id": replicate_id, "simulation_seed": sim_seed,
            "cases": cases * 100_000, "deaths": deaths * 100_000, "doses": doses,
        })
    return rows


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--network-cache", type=Path, default=(
        REPO / "data_and_results/network_cache/production_network_20260722"
        / "network_000_seed_2026072501.npz"))
    parser.add_argument("--output-dir", type=Path, default=(
        REPO / "data_and_results/outputs/s4_independent_ve_fine_20260723"))
    parser.add_argument("--smoke", action="store_true",
                        help="3 replicates x 2 probe cells + base arm, no summary/manifest")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not args.network_cache.exists():
        raise FileNotFoundError(f"Network cache not found: {args.network_cache}")

    fitted_path = REPO / "data_and_results" / "fitted_parameters.json"
    params = json.loads(fitted_path.read_text())

    if args.smoke:
        tasks = [(None, None, 3), (0.0, 0.0, 3), (0.0, 0.9, 3), (0.9, 0.9, 3)]
    else:
        tasks = [(None, None, args.replicates)]
        tasks += [(ve_i, ve_m, args.replicates) for ve_i in VE_VALUES for ve_m in VE_VALUES]

    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "s4_raw.csv"
    cache_path = str(args.network_cache)
    with raw_path.open("w", newline="") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=FIELDS)
        writer.writeheader()
        if args.workers == 1:
            initialise_worker(params, cache_path)
            for result in map(run_cell, tasks):
                writer.writerows(result)
        else:
            with mp.Pool(args.workers, initializer=initialise_worker,
                         initargs=(params, cache_path)) as pool:
                for result in pool.imap_unordered(run_cell, tasks):
                    writer.writerows(result)

    if args.smoke:
        print(f"Smoke output written to {raw_path}")
        return

    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    base_deaths = {int(r["replicate_id"]): float(r["deaths"])
                   for r in rows if r["cell"] == "base_no_vax"}
    summary = []
    for ve_i in VE_VALUES:
        for ve_m in VE_VALUES:
            cell = f"vei_{ve_i:.3f}_vem_{ve_m:.3f}"
            group = [r for r in rows if r["cell"] == cell]
            deaths = np.array([float(r["deaths"]) for r in group])
            reductions = np.array([
                100.0 * (base_deaths[int(r["replicate_id"])] - float(r["deaths"]))
                / base_deaths[int(r["replicate_id"])] for r in group])
            summary.append({
                "cell": cell, "ve_i": f"{ve_i:.3f}", "ve_m": f"{ve_m:.3f}",
                "n_replicates": len(group),
                "median_cases": float(np.median([float(r["cases"]) for r in group])),
                "median_deaths": float(np.median(deaths)),
                "median_doses": float(np.median([float(r["doses"]) for r in group])),
                "median_mortality_reduction_pct": float(np.median(reductions)),
            })
    base_vals = np.array(list(base_deaths.values()))
    summary.append({
        "cell": "base_no_vax", "ve_i": "", "ve_m": "",
        "n_replicates": len(base_vals), "median_cases": "",
        "median_deaths": float(np.median(base_vals)), "median_doses": "",
        "median_mortality_reduction_pct": 0.0,
    })
    summary_path = args.output_dir / "s4_grid_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "grid": "13x13 independent efficacy; ve_i and ve_m in 0, 0.075, ..., 0.90",
            "replicates_per_cell": args.replicates,
            "seed_schedule": f"{SEED_BASE}+replicate_id, paired across all cells and base arm",
            "scenario": "community vaccination, 40% coverage, 14-day rollout from day 0, base operations",
            "efficacy_mapping": "efficacy=ve_i (infection path, sigmoidal ramp k=0.5 d0=10); "
                                "vax_CFR=base_CFR*(1-ve_m) (mortality path); "
                                "infection_efficacy_multiplier=1.0",
            "base_arm": "base operations, no community vaccination, efficacy 0",
        },
        "model": {
            "incubation_period": 8.5, "infectious_period": 6.0,
            "incubation_shape": 2.0, "infectious_shape": 2.0,
            "sigmoidal_k": 0.5, "sigmoidal_d0": 10.0,
            "initial_infected": 15, "initial_exposed": 15, "max_sim_time": HORIZON,
            "detection_delay": 4.0, "tracing_delay": 2.0,
            "reporting_rate": 0.30, "tracing_coverage": 0.30,
            "max_daily_traces": 100, "uptake": 0.8, "vaccine_acceptability": 1.0,
            "allow_pep": True, "base_CFR": params["base_CFR"],
        },
        "network_cache": str(args.network_cache),
        "network_cache_sha256": sha256(args.network_cache),
        "cpp_sha256": sha256(ROOT / "ebola_stochastic_ring_cpp.cpp"),
        "wrapper_sha256": sha256(ROOT / "ebola_stochastic_ring.py"),
        "fitted_parameters_sha256": sha256(fitted_path),
        "workers": args.workers,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {raw_path}, {summary_path}, manifest.json")


if __name__ == "__main__":
    run()
