"""Paired simulation-horizon extension analysis.

Quantifies outcome censoring from truncating simulations at day 90: key
strategies are run at 90-, 180-, and 365-day horizons with matched seeds.
Because the event stream is independent of the cutoff, a longer-horizon run
with the same seed is an exact continuation of its 90-day counterpart, so
within-replicate differences isolate post-day-90 accrual.  Rt is held at its
final estimated value beyond the estimation window, as in the primary
analyses; longer horizons therefore represent an upper bound on censoring,
since no further decline in transmission is assumed.
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
FITTED_PATH = REPO / "data_and_results" / "fitted_parameters.json"
DEFAULT_CACHE = (REPO / "data_and_results" / "network_cache"
                 / "production_network_clustered_20260803" / "network_000_seed_2026080301.npz")
BASE_SEED = 2026080400
HORIZONS = (90, 180)
FIELDS = ["strategy", "horizon", "replicate_id", "simulation_seed",
          "cases", "deaths", "doses"]
WORKER_GRAPH = None
WORKER_PARAMS = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ramp(target, duration, horizon):
    return np.linspace(0.30, target, duration).tolist() + [target] * (horizon + 1 - duration)


def build_scenarios(horizon):
    base_report = [0.30] * (horizon + 1)
    base_trace = [0.30] * (horizon + 1)
    base_ops = {
        "reporting_rate": base_report, "tracing_coverage": base_trace,
        "detection_delay": 4.0, "tracing_delay": 2.0, "ring_radius": 2,
        "efficacy": 0.0, "max_vaccines": 0,
        "community_vax_coverage": 0.0, "community_vax_trigger": 0,
        "community_vax_delay": -1.0, "community_vax_rollout_days": 0.0,
    }
    enhanced_ops = {
        **base_ops,
        "reporting_rate": ramp(0.70, 15, horizon),
        "tracing_coverage": ramp(0.80, 15, horizon),
        "detection_delay": 2.0, "tracing_delay": 1.0,
    }
    scenarios = {
        "no_vax_base_ops": base_ops,
        "no_vax_enh_ops": enhanced_ops,
        "vax_base_ops": {**base_ops, "efficacy": 0.45, "max_vaccines": None},
        "vax_enh_ops": {**enhanced_ops, "efficacy": 0.45, "max_vaccines": None},
    }
    for coverage in (0.40, 0.60):
        scenarios[f"comm_base_{int(coverage * 100)}"] = {
            **base_ops, "efficacy": 0.45, "max_vaccines": 0,
            "community_vax_coverage": coverage, "community_vax_trigger": 1,
            "community_vax_delay": 0.0, "community_vax_rollout_days": 14.0,
        }
    return scenarios


WORKER_ACCEPTANCE = 0.90


def initialise_worker(params, network_cache_path, acceptance):
    global WORKER_GRAPH, WORKER_PARAMS, WORKER_ACCEPTANCE
    from network_cache import load_cached_network
    WORKER_GRAPH = load_cached_network(network_cache_path)
    WORKER_PARAMS = params
    WORKER_ACCEPTANCE = acceptance


def run_replicate(replicate_id):
    params = WORKER_PARAMS
    rt = list(params["Rt_array"])
    rt.extend([rt[-1]] * max(0, max(HORIZONS) + 1 - len(rt)))
    sim_seed = BASE_SEED + replicate_id
    rows = []
    for horizon in HORIZONS:
        for name, sc in build_scenarios(horizon).items():
            cases, deaths, doses, _ = simulate_ring_vaccination(
                WORKER_GRAPH, rt_array=rt, baseline_tau=0.25,
                incubation_period=8.5, infectious_period=6.0,
                ring_radius=sc["ring_radius"], efficacy=sc["efficacy"],
                immune_delay=10.0, uptake=0.8,
                reporting_rate=sc["reporting_rate"],
                tracing_coverage=sc["tracing_coverage"],
                vaccine_acceptability=WORKER_ACCEPTANCE,
                detection_delay=sc["detection_delay"],
                tracing_delay=sc["tracing_delay"], max_daily_traces=100,
                max_vaccines=sc["max_vaccines"],
                base_CFR=float(params["base_CFR"]),
                initial_infected=15, initial_exposed=15,
                max_sim_time=horizon, seed=sim_seed, engine="cpp",
                community_vax_coverage=sc["community_vax_coverage"],
                community_vax_trigger=sc["community_vax_trigger"],
                community_vax_delay=sc["community_vax_delay"],
                community_vax_rollout_days=sc["community_vax_rollout_days"],
                sigmoidal_k=0.5, sigmoidal_d0=10.0, allow_pep=True,
                incubation_shape=2.0, infectious_shape=2.0,
            )
            n = WORKER_GRAPH.number_of_nodes()
            rows.append({"strategy": name, "horizon": horizon,
                         "replicate_id": replicate_id, "simulation_seed": sim_seed,
                         "cases": cases * n, "deaths": deaths * n, "doses": doses})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--network-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--acceptance", type=float, default=0.90,
                        help="Conditional vaccine acceptance (base case 0.90)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not args.network_cache.exists():
        raise FileNotFoundError(f"Network cache not found: {args.network_cache}")

    params = json.loads(FITTED_PATH.read_text())
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "horizon_extension_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with mp.Pool(args.workers, initializer=initialise_worker,
                     initargs=(params, str(args.network_cache), args.acceptance)) as pool:
            for rows in pool.imap_unordered(run_replicate, range(args.replicates)):
                writer.writerows(rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicates": args.replicates, "horizons": list(HORIZONS),
        "seed_schedule": f"{BASE_SEED}+replicate; same seed across horizons (exact continuation) "
                         "and across strategies (paired)",
        "rt_extension": "Rt held at final estimated value beyond the estimation window",
        "network_cache": str(args.network_cache),
        "network_cache_sha256": sha256(args.network_cache),
        "base_parameters": {
            "incubation_period": 8.5, "infectious_period": 6.0,
            "incubation_shape": 2.0, "infectious_shape": 2.0,
            "base_CFR": float(params["base_CFR"]), "efficacy": 0.45,
            "sigmoidal_k": 0.5, "sigmoidal_d0": 10.0,
            "initial_infected": 15, "initial_exposed": 15,
            "max_daily_traces": 100, "uptake": 0.8,
            "vaccine_acceptability": args.acceptance, "allow_pep": True,
            "baseline_tau": 0.25, "ring_radius": 2,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {raw_path} and manifest.json")


if __name__ == "__main__":
    main()
