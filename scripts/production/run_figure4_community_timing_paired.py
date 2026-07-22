"""Same-seed community-vaccination coverage-by-delay grid for Figure 4."""
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
G = P = RT = None
N = 100_000


def init(params, rt, cache):
    global G, P, RT
    G, P, RT = load_cached_network(cache), params, rt


def sim(coverage, delay, seed):
    return simulate_ring_vaccination(
        G, rt_array=RT, incubation_period=8.5, infectious_period=6.0,
        ring_radius=2, efficacy=1.0, infection_efficacy_multiplier=.45,
        vax_CFR=float(P["base_CFR"]) * .55, uptake=.8,
        reporting_rate=[.30] * 91, tracing_coverage=.30, vaccine_acceptability=1.0,
        detection_delay=4.0, tracing_delay=2.0, max_daily_traces=100,
        max_vaccines=0, base_CFR=float(P["base_CFR"]), initial_infected=15,
        initial_exposed=15, max_sim_time=90, seed=seed, engine="cpp",
        sigmoidal_k=.5, sigmoidal_d0=10.0, allow_pep=True,
        community_vax_coverage=coverage, community_vax_trigger=1 if coverage else 0,
        community_vax_delay=delay if coverage else -1.0,
        community_vax_rollout_days=14.0 if coverage else 0.0,
        incubation_shape=2.0, infectious_shape=2.0,
    )


def task(task_spec):
    coverage, delay, reps, seed0 = task_spec
    rows = []
    for replicate in range(reps):
        seed = seed0 + replicate
        base_deaths = sim(0.0, -1.0, seed)[1] * N
        scenario_deaths = sim(coverage, delay, seed)[1] * N
        rows.append({"coverage": coverage, "delay": delay, "replicate": replicate, "seed": seed,
                     "base_deaths": base_deaths, "scenario_deaths": scenario_deaths,
                     "mortality_reduction_pct": 100 * (base_deaths - scenario_deaths) / base_deaths})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    params = json.loads((REPO / "data_and_results/fitted_parameters.json").read_text())
    rt = list(params["Rt_array"]) + [params["Rt_array"][-1]] * 25
    specs = [(float(coverage), float(delay), args.replicates, 2026900000 + i * 1000)
             for i, delay in enumerate((0, 14, 28)) for coverage in np.linspace(.1, .8, 8)]
    rows = []
    with mp.Pool(args.workers, initializer=init, initargs=(params, rt, str(args.network_cache))) as pool:
        for result in pool.imap_unordered(task, specs):
            rows.extend(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "delays_days": [0, 14, 28], "coverage": [round(x, 1) for x in np.linspace(.1, .8, 8)],
        "replicates": args.replicates, "comparison": "same-seed base operations without vaccination"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
