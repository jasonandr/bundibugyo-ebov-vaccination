"""Paired early-extinction analysis for the manuscript.

Four strategies are evaluated with matched simulation seeds on the cached
100,000-person production network.  Each replicate uses the C++ engine with
``return_mechanism=True`` so that per-person exposure/onset/removal times are
available.  A replicate is classed as extinct before day 90 when every
ever-infected person has a recorded removal event (recovery_or_death_time set,
i.e. not the -1 sentinel; seed events are legitimately negative because the
initial infectious seeds onset at -detection_delay) and the last removal occurs
before day 90.

Seeds are matched across strategies: replicate i uses simulation seed
2026073000 + i for every strategy.
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
BASE_SEED = 2026073000
NETWORK_CACHE = REPO / "data_and_results" / "network_cache" / "production_network_20260722" / "network_000_seed_2026072501.npz"
FITTED_PATH = REPO / "data_and_results" / "fitted_parameters.json"
FIELDS = [
    "strategy", "replicate_id", "simulation_seed", "extinct_before_day90",
    "extinction_time", "total_deaths", "total_infected", "total_vaccines",
]

WORKER_GRAPH = None
WORKER_PARAMS = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ramp(target, duration=15):
    """Identical to run_pooled_psa_figure2_pilot.ramp (length HORIZON + 1)."""
    return np.linspace(0.30, target, duration).tolist() + [target] * (HORIZON + 1 - duration)


def build_scenarios():
    base_report = [0.30] * (HORIZON + 1)
    base_trace = [0.30] * (HORIZON + 1)
    base_ops = {
        "reporting_rate": base_report, "tracing_coverage": base_trace,
        "detection_delay": 4.0, "tracing_delay": 2.0, "ring_radius": 2,
        "efficacy": 0.0, "max_vaccines": 0,
        "community_vax_coverage": 0.0, "community_vax_trigger": 0,
        "community_vax_delay": -1.0, "community_vax_rollout_days": 0.0,
    }
    enhanced_ops = {
        **base_ops,
        "reporting_rate": ramp(0.70), "tracing_coverage": ramp(0.80),
        "detection_delay": 2.0, "tracing_delay": 1.0,
    }
    scenarios = {
        "no_vax_base_ops": base_ops,
        "no_vax_enh_ops": enhanced_ops,
    }
    for coverage in (0.40, 0.60):
        scenarios[f"comm_base_{int(coverage * 100)}"] = {
            **base_ops,
            "efficacy": 0.45,
            "community_vax_coverage": coverage,
            "community_vax_trigger": 1,
            "community_vax_delay": 0.0,
            "community_vax_rollout_days": 14.0,
        }
    return scenarios


def initialise_worker(params, network_cache_path):
    """Load the cached network and fitted parameters once per worker."""
    global WORKER_GRAPH, WORKER_PARAMS
    from network_cache import load_cached_network
    WORKER_GRAPH = load_cached_network(network_cache_path)
    WORKER_PARAMS = params


def extinction_metrics(res):
    exposure = np.asarray(res["exposure_time"], dtype=float)
    onset = np.asarray(res["onset_time"], dtype=float)
    removal = np.asarray(res["recovery_or_death_time"], dtype=float)
    # Ever-infected: exposure event processed, or an onset recorded.  The 15
    # initial infectious seeds have no EXPOSURE event (they onset directly at
    # -detection_delay), so the onset check is required to include them.
    infected = (exposure >= 0.0) | (onset != -1.0)
    removal_times = removal[infected]
    # -1.0 is the documented "unset" sentinel.  Resolved seed events can be
    # negative (initial infectious seeds onset at -detection_delay), so the
    # resolution test is `!= -1` rather than `>= 0`.
    resolved = removal_times != -1.0
    extinct = bool(resolved.all()) and removal_times.size > 0 and float(removal_times.max()) < HORIZON
    extinction_time = float(removal_times.max()) if extinct else float("nan")
    return extinct, extinction_time


def run_replicate(replicate_id):
    params = WORKER_PARAMS
    rt = list(params["Rt_array"])
    rt.extend([rt[-1]] * max(0, HORIZON + 1 - len(rt)))
    sim_seed = BASE_SEED + replicate_id
    rows = []
    for name, sc in build_scenarios().items():
        res = simulate_ring_vaccination(
            WORKER_GRAPH, rt_array=rt, baseline_tau=0.25,
            incubation_period=8.5, infectious_period=6.0,
            ring_radius=sc["ring_radius"], efficacy=sc["efficacy"], uptake=0.8,
            reporting_rate=sc["reporting_rate"], tracing_coverage=sc["tracing_coverage"],
            vaccine_acceptability=1.0, detection_delay=sc["detection_delay"],
            tracing_delay=sc["tracing_delay"], max_daily_traces=100,
            max_vaccines=sc["max_vaccines"], base_CFR=float(params["base_CFR"]),
            initial_infected=15, initial_exposed=15, max_sim_time=HORIZON,
            seed=sim_seed, engine="cpp",
            community_vax_coverage=sc["community_vax_coverage"],
            community_vax_trigger=sc["community_vax_trigger"],
            community_vax_delay=sc["community_vax_delay"],
            community_vax_rollout_days=sc["community_vax_rollout_days"],
            sigmoidal_k=0.5, sigmoidal_d0=10.0, allow_pep=True,
            incubation_shape=2.0, infectious_shape=2.0,
            return_mechanism=True,
        )
        extinct, extinction_time = extinction_metrics(res)
        rows.append({
            "strategy": name, "replicate_id": replicate_id, "simulation_seed": sim_seed,
            "extinct_before_day90": int(extinct), "extinction_time": extinction_time,
            "total_deaths": int(res["total_deaths"]), "total_infected": int(res["total_infected"]),
            "total_vaccines": int(res["total_vaccines"]),
        })
    return rows


def wilson_interval(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def summarize(rows):
    summary = []
    for strategy in build_scenarios():
        group = [r for r in rows if r["strategy"] == strategy]
        n = len(group)
        extinct_times = np.array([r["extinction_time"] for r in group if r["extinct_before_day90"]], dtype=float)
        deaths = np.array([r["total_deaths"] for r in group], dtype=float)
        k = int(extinct_times.size)
        low, high = wilson_interval(k, n)
        summary.append({
            "strategy": strategy, "n_replicates": n, "n_extinct": k,
            "fraction_extinct": k / n,
            "wilson95_low": low, "wilson95_high": high,
            "median_extinction_time": float(np.median(extinct_times)) if k else float("nan"),
            "extinction_time_q25": float(np.percentile(extinct_times, 25)) if k else float("nan"),
            "extinction_time_q75": float(np.percentile(extinct_times, 75)) if k else float("nan"),
            "median_deaths": float(np.median(deaths)),
            "deaths_q25": float(np.percentile(deaths, 25)),
            "deaths_q75": float(np.percentile(deaths, 75)),
        })
    return summary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "data_and_results" / "review_outputs" / "extinction_paired_20260723")
    args = parser.parse_args()
    if args.replicates < 1 or args.workers < 1:
        raise ValueError("Use at least one replicate and one worker")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not NETWORK_CACHE.exists():
        raise FileNotFoundError(f"Network cache not found: {NETWORK_CACHE}")

    params = json.loads(FITTED_PATH.read_text())
    args.output_dir.mkdir(parents=True)

    raw_path = args.output_dir / "extinction_paired_raw.csv"
    started = datetime.now(timezone.utc)
    with raw_path.open("w", newline="") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=FIELDS)
        writer.writeheader()
        tasks = range(args.replicates)
        if args.workers == 1:
            initialise_worker(params, str(NETWORK_CACHE))
            result_iter = map(run_replicate, tasks)
            for result in result_iter:
                writer.writerows(result)
        else:
            with mp.Pool(args.workers, initializer=initialise_worker,
                         initargs=(params, str(NETWORK_CACHE))) as pool:
                done = 0
                for result in pool.imap_unordered(run_replicate, tasks):
                    writer.writerows(result)
                    done += 1
                    if done % 50 == 0:
                        print(f"[{datetime.now(timezone.utc).isoformat()}] {done}/{args.replicates} replicates complete", flush=True)

    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["extinct_before_day90"] = int(row["extinct_before_day90"])
        row["extinction_time"] = float(row["extinction_time"])
        row["total_deaths"] = int(row["total_deaths"])
        row["total_infected"] = int(row["total_infected"])
    summary = summarize(rows)
    summary_path = args.output_dir / "extinction_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started.isoformat(),
        "design": {
            "strategies": list(build_scenarios()),
            "replicates_per_strategy": args.replicates,
            "base_seed": BASE_SEED,
            "seed_schedule": f"{BASE_SEED}+i for replicate i, matched across strategies",
            "workers": args.workers,
            "horizon_days": HORIZON,
            "extinction_definition": (
                "all ever-infected persons have recovery_or_death_time != -1 (unset sentinel; "
                "negative seed removal times are resolved events) and max(recovery_or_death_time) < 90"
            ),
            "base_parameters": {
                "incubation_mean": 8.5, "infectious_mean": 6.0,
                "incubation_shape": 2.0, "infectious_shape": 2.0,
                "base_CFR": float(params["base_CFR"]), "vaccine_efficacy_community": 0.45,
                "sigmoidal_k": 0.5, "sigmoidal_d0": 10.0,
                "initial_infected": 15, "initial_exposed": 15,
                "max_daily_traces": 100, "uptake": 0.8,
                "vaccine_acceptability": 1.0, "allow_pep": True,
                "baseline_tau": 0.25,
            },
            "scenarios": {name: {k: (v if not isinstance(v, list) else f"array len {len(v)}: {v[0]}..{v[-1]}")
                                   for k, v in sc.items()}
                          for name, sc in build_scenarios().items()},
        },
        "network": {
            "cache_path": str(NETWORK_CACHE),
            "network_cache_sha256": sha256(NETWORK_CACHE),
        },
        "fitted_parameters_path": str(FITTED_PATH),
        "fitted_parameters_sha256": sha256(FITTED_PATH),
        "cpp_sha256": sha256(ROOT / "ebola_stochastic_ring_cpp.cpp"),
        "wrapper_sha256": sha256(ROOT / "ebola_stochastic_ring.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "engine": "cpp (daily pooled onset-cohort allocator, use_cohort=True), return_mechanism=True",
        "outputs": {"raw": str(raw_path), "summary": str(summary_path)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {raw_path}, {summary_path}, manifest.json")


if __name__ == "__main__":
    run()
