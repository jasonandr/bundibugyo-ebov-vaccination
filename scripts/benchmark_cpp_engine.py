"""Small, reproducible benchmark for the stochastic C++ backend."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from ebola_stochastic_ring import generate_network, simulate_ring_vaccination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()

    np.random.seed(20260716)
    t0 = time.perf_counter()
    graph = generate_network(args.population)
    graph_seconds = time.perf_counter() - t0

    rt = [1.8] * (args.horizon + 1)
    run_times = []
    results = []
    for replicate in range(args.replicates):
        start = time.perf_counter()
        result = simulate_ring_vaccination(
            graph,
            rt_array=rt,
            baseline_tau=0.22,
            incubation_period=8.5,
            infectious_period=6.0,
            ring_radius=args.radius,
            vaccine_effect=0.45,
            reporting_rate=0.7,
            tracing_coverage=0.8,
            vaccine_acceptability=0.8,
            detection_delay=4.0,
            max_daily_traces=1000,
            base_CFR=0.454,
            initial_infected=5,
            max_sim_time=args.horizon,
            sigmoidal_k=0.8,
            sigmoidal_d0=10.0,
            engine="cpp",
            seed=1000 + replicate,
        )
        run_times.append(time.perf_counter() - start)
        results.append([float(x) for x in result])

    report = {
        "population": args.population,
        "replicates": args.replicates,
        "horizon": args.horizon,
        "radius": args.radius,
        "graph_seconds": graph_seconds,
        "simulation_seconds_total": sum(run_times),
        "seconds_per_replicate_mean": float(np.mean(run_times)),
        "seconds_per_replicate_median": float(np.median(run_times)),
        "seconds_per_replicate_p95": float(np.quantile(run_times, 0.95)),
        "results_mean": np.mean(np.asarray(results), axis=0).tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
