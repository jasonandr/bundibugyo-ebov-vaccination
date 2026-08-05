"""Corrected, non-destructive PSA pilot for Table 2/Figure 2 values.

Each Latin-hypercube draw is evaluated with matched stochastic replicates on a
single fixed 100,000-person 5.2/30/160 network.  The output estimand is an
expected outcome within parameter draw; uncertainty intervals are percentiles
across parameter-draw expected values.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

from ebola_stochastic_ring import generate_network, simulate_ring_vaccination


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
N = 100_000
HORIZON = 90
NETWORK_SEED = 2026072501
FIELDS = [
    "draw_id", "replicate_id", "simulation_seed", "scenario", "vaccine_efficacy",
    "rt_posterior_index", "incubation_mean", "infectious_mean", "detection_delay_base",
    "detection_delay_enhanced", "tracing_coverage_enhanced", "incubation_shape",
    "infectious_shape", "vaccine_acceptance", "cases", "deaths", "doses",
]
WORKER_GRAPH = None
WORKER_PARAMS = None
WORKER_POSTERIOR = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ramp(target, duration=15):
    return np.linspace(0.30, target, duration).tolist() + [target] * (HORIZON + 1 - duration)


def lhs_draws(n_draws, n_rt_samples, seed, acceptance_seed=2026080401):
    """Archived PSA distributions, excluding the previously unused variance draw.

    Conditional vaccine acceptance is drawn from an independent 1-D
    Latin-hypercube sequence (Uniform 0.75-0.95) so that the 9-dimensional
    draw matrix is unchanged and runs remain paired with configurations that
    fix acceptance at 1.0.
    """
    unit = qmc.LatinHypercube(d=9, seed=seed).random(n=n_draws)
    acc_unit = qmc.LatinHypercube(d=1, seed=acceptance_seed).random(n=n_draws)
    acceptance = qmc.scale(acc_unit, [0.75], [0.95])
    return [
        {
            "draw_id": i,
            "vaccine_efficacy": float(qmc.scale(unit[i:i + 1, 0:1], [0.30], [0.60])[0, 0]),
            "rt_posterior_index": min(n_rt_samples - 1, int(unit[i, 1] * n_rt_samples)),
            "incubation_mean": float(norm.ppf(0.05 + 0.90 * unit[i, 2], loc=8.5, scale=1.0)),
            "infectious_mean": float(norm.ppf(0.05 + 0.90 * unit[i, 3], loc=6.0, scale=0.8)),
            "detection_delay_base": float(qmc.scale(unit[i:i + 1, 4:5], [3.0], [5.0])[0, 0]),
            "detection_delay_enhanced": float(qmc.scale(unit[i:i + 1, 5:6], [1.5], [3.5])[0, 0]),
            "tracing_coverage_enhanced": float(qmc.scale(unit[i:i + 1, 6:7], [0.60], [0.90])[0, 0]),
            "incubation_shape": float(qmc.scale(unit[i:i + 1, 7:8], [1.0], [3.0])[0, 0]),
            "infectious_shape": float(qmc.scale(unit[i:i + 1, 8:9], [1.0], [3.0])[0, 0]),
            "vaccine_acceptance": float(acceptance[i, 0]),
        }
        for i in range(n_draws)
    ]


def scenarios(draw, include_ring1=False):
    base_report, base_trace = [0.30] * (HORIZON + 1), [0.30] * (HORIZON + 1)
    enhanced_report = ramp(0.70)
    enhanced_trace = ramp(draw["tracing_coverage_enhanced"])
    ve = draw["vaccine_efficacy"]
    baseline = {"report": base_report, "trace": base_trace, "delay": draw["detection_delay_base"],
                "trace_delay": 2.0, "radius": 2, "ve": 0.0, "max_vaccines": 0, "coverage": 0.0}
    enhanced = {"report": enhanced_report, "trace": enhanced_trace, "delay": draw["detection_delay_enhanced"],
                "trace_delay": 1.0, "radius": 2, "ve": 0.0, "max_vaccines": 0, "coverage": 0.0}
    out = {
        "no_vax_base_ops": baseline,
        "vax_base_ops": {**baseline, "ve": ve, "max_vaccines": None},
        "no_vax_enh_ops": enhanced,
        "vax_enh_ops": {**enhanced, "ve": ve, "max_vaccines": None},
    }
    if include_ring1:
        # These are retained separately from the Figure 2 strategy set.  They
        # use the identical parameter draw, cached network, and seed schedule
        # as Ring 2, so the dose-efficiency comparison is matched throughout.
        # `ring_radius` also controls the reach of monitoring/contact tracing,
        # so Ring 1 vaccination must be compared with a Radius 1 no-vaccine
        # arm rather than the Radius 2 primary comparator.
        out["no_vax_ring1_base_ops"] = {**baseline, "radius": 1}
        out["no_vax_ring1_enh_ops"] = {**enhanced, "radius": 1}
        out["ring1_vax_base_ops"] = {**baseline, "radius": 1, "ve": ve, "max_vaccines": None}
        out["ring1_vax_enh_ops"] = {**enhanced, "radius": 1, "ve": ve, "max_vaccines": None}
    for coverage in (0.2, 0.4, 0.6, 0.8):
        out[f"comm_base_{int(coverage * 100)}"] = {
            **baseline, "ve": ve, "coverage": coverage, "max_vaccines": 0,
        }
    return out


def initialise_worker(params, posterior, network_cache_path=None):
    """Build one identical fixed network per worker, then reuse it across draws."""
    global WORKER_GRAPH, WORKER_PARAMS, WORKER_POSTERIOR
    if network_cache_path:
        from network_cache import load_cached_network
        WORKER_GRAPH = load_cached_network(network_cache_path)
    else:
        np.random.seed(NETWORK_SEED)
        WORKER_GRAPH = generate_network(N, household_mean=5.2, community_mean=30.0, community_variance=160.0)
    WORKER_PARAMS = params
    WORKER_POSTERIOR = posterior


def run_draw(task):
    draw, replicates, include_ring1 = task
    rt = WORKER_POSTERIOR[draw["rt_posterior_index"]].astype(float).tolist()
    rt.extend([rt[-1]] * max(0, HORIZON + 1 - len(rt)))
    rows = []
    for replicate_id in range(replicates):
        sim_seed = 2026072700 + draw["draw_id"] * 10_000 + replicate_id
        for name, sc in scenarios(draw, include_ring1=include_ring1).items():
            cases, deaths, doses, _ = simulate_ring_vaccination(
                WORKER_GRAPH, rt_array=rt, baseline_tau=float(WORKER_PARAMS.get("baseline_tau", 0.25)),
                incubation_period=draw["incubation_mean"], infectious_period=draw["infectious_mean"],
                ring_radius=sc["radius"], efficacy=sc["ve"], uptake=0.8,
                reporting_rate=sc["report"], tracing_coverage=sc["trace"],
                vaccine_acceptability=draw.get("vaccine_acceptance", 1.0), detection_delay=sc["delay"],
                tracing_delay=sc["trace_delay"], max_daily_traces=100,
                max_vaccines=sc["max_vaccines"], base_CFR=float(WORKER_PARAMS["base_CFR"]),
                initial_infected=15, initial_exposed=15, max_sim_time=HORIZON,
                seed=sim_seed, engine="cpp", community_vax_coverage=sc["coverage"],
                community_vax_trigger=1 if sc["coverage"] else 0,
                community_vax_delay=0.0 if sc["coverage"] else -1.0,
                community_vax_rollout_days=14.0 if sc["coverage"] else 0.0,
                sigmoidal_k=0.5, sigmoidal_d0=10.0, allow_pep=True,
                incubation_shape=draw["incubation_shape"], infectious_shape=draw["infectious_shape"],
            )
            rows.append({**draw, "replicate_id": replicate_id, "simulation_seed": sim_seed,
                         "scenario": name, "cases": cases * N, "deaths": deaths * N, "doses": doses})
    return rows


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--network-cache", type=Path)
    parser.add_argument("--include-ring1", action="store_true",
                        help="Add matched Ring 1 base/enhanced strategies for the dose-efficiency analysis.")
    parser.add_argument("--rt-source", choices=("fitted-median", "posterior"), default="fitted-median",
                        help="Use the updated EpiNow2 median by default; posterior is permitted only for a verified matching draw file.")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--fixed-acceptance", type=float, default=None,
                        help="If set, override the per-draw LHS acceptance draw with this fixed value.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data_and_results/pooled_psa_figure2_pilot_20260722"))
    args = parser.parse_args()
    if args.draws < 2 or args.replicates < 1 or args.workers < 1:
        raise ValueError("Use at least two draws, one replicate, and one worker")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.network_cache and not args.network_cache.exists():
        raise FileNotFoundError(f"Network cache not found: {args.network_cache}")

    fitted_path = REPO / "data_and_results" / "fitted_parameters.json"
    params = json.loads(fitted_path.read_text())
    posterior_path = REPO / "data_and_results" / "rt_posterior_samples.npy"
    if args.rt_source == "fitted-median":
        posterior = np.asarray([params["Rt_array"]], dtype=float)
    else:
        posterior = np.load(posterior_path)
        if posterior.ndim != 2 or posterior.shape[0] < 2:
            raise ValueError("rt_posterior_samples.npy must contain multiple Rt trajectories")

    args.output_dir.mkdir(parents=True)
    draws = lhs_draws(args.draws, posterior.shape[0], args.seed)
    if args.fixed_acceptance is not None:
        for draw in draws:
            draw["vaccine_acceptance"] = float(args.fixed_acceptance)
    raw_path = args.output_dir / "raw_replicates.csv"
    with raw_path.open("w", newline="") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=FIELDS)
        writer.writeheader()
        tasks = [(draw, args.replicates, args.include_ring1) for draw in draws]
        cache_path = str(args.network_cache) if args.network_cache else None
        if args.workers == 1:
            initialise_worker(params, posterior, cache_path)
            result_iter = map(run_draw, tasks)
            for result in result_iter:
                writer.writerows(result)
        else:
            with mp.Pool(args.workers, initializer=initialise_worker, initargs=(params, posterior, cache_path)) as pool:
                for result in pool.imap_unordered(run_draw, tasks):
                    writer.writerows(result)

    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["draw_id"]), row["scenario"]), []).append(row)
    draw_values = []
    for (draw_id, scenario), group in grouped.items():
        base = next(r for r in rows if int(r["draw_id"]) == draw_id and r["scenario"] == "no_vax_base_ops")
        base_group = grouped[(draw_id, "no_vax_base_ops")]
        enhanced_group = grouped[(draw_id, "no_vax_enh_ops")]
        mean_deaths = float(np.mean([float(r["deaths"]) for r in group]))
        mean_cases = float(np.mean([float(r["cases"]) for r in group]))
        mean_doses = float(np.mean([float(r["doses"]) for r in group]))
        # Every named strategy is reported versus base operations.  The one
        # exception is the separately appended incremental-ring estimand.
        comparator_group = base_group
        if scenario.startswith("ring1_") or scenario == "no_vax_ring1_enh_ops":
            comparator_group = grouped[(draw_id, "no_vax_ring1_base_ops")]
        comparator_deaths = float(np.mean([float(r["deaths"]) for r in comparator_group]))
        draw_values.append({"draw_id": draw_id, "scenario": scenario, "mean_cases": mean_cases,
                            "mean_deaths": mean_deaths, "mean_doses": mean_doses,
                            "comparator_mean_deaths": comparator_deaths,
                            "deaths_averted_pct": 100 * (comparator_deaths - mean_deaths) / comparator_deaths})
        if scenario == "vax_enh_ops":
            draw_values.append({"draw_id": draw_id, "scenario": "incremental_ring_vax",
                                "mean_cases": mean_cases, "mean_deaths": mean_deaths,
                                "mean_doses": mean_doses,
                                "comparator_mean_deaths": float(np.mean([float(r["deaths"]) for r in enhanced_group])),
                                "deaths_averted_pct": 100 * (float(np.mean([float(r["deaths"]) for r in enhanced_group])) - mean_deaths) /
                                    float(np.mean([float(r["deaths"]) for r in enhanced_group]))})

    draw_path = args.output_dir / "draw_expected_values.csv"
    with draw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(draw_values[0]))
        writer.writeheader(); writer.writerows(draw_values)
    summary = []
    for scenario in sorted({r["scenario"] for r in draw_values}):
        values = np.array([r["deaths_averted_pct"] for r in draw_values if r["scenario"] == scenario])
        summary.append({"scenario": scenario, "n_draws": len(values),
                        "median_deaths_averted_pct": float(np.median(values)),
                        "ui_low_95": float(np.quantile(values, 0.025)),
                        "ui_high_95": float(np.quantile(values, 0.975)),
                        "mean_deaths_averted_pct": float(np.mean(values))})
    with (args.output_dir / "figure2_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    cache_manifest = None
    if args.network_cache:
        sidecar = args.network_cache.with_suffix(".manifest.json")
        if sidecar.exists():
            cache_manifest = json.loads(sidecar.read_text())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "draws": args.draws,
        "replicates_per_draw": args.replicates, "workers": args.workers,
        "total_runs_per_strategy": args.draws * args.replicates,
        "include_ring1": args.include_ring1,
        "network": {"N": N, "household_mean": 5.2, "community_mean": 30.0, "community_variance": 160.0,
                    "network_seed": NETWORK_SEED, "cache_path": str(args.network_cache) if args.network_cache else None,
                    "cache_manifest": cache_manifest},
        "allocator": "daily pooled onset cohort", "waiting_times": "Gamma; LHS shapes Uniform(1,3)",
        "rt_source": args.rt_source,
        "vaccine_acceptance": (f"fixed at {args.fixed_acceptance}" if args.fixed_acceptance is not None
                               else "LHS draw, Uniform(0.75, 0.95), independent 1-D sequence seed 2026080401"),
        "cpp_sha256": sha256(ROOT / "ebola_stochastic_ring_cpp.cpp"),
        "wrapper_sha256": sha256(ROOT / "ebola_stochastic_ring.py"),
        "fitted_parameters_sha256": sha256(fitted_path),
        "rt_posterior_sha256": sha256(posterior_path) if args.rt_source == "posterior" else None,
        "note": "Pilot only. Community-degree variance is held at 160 because network topology is fixed within this pilot.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    run()
