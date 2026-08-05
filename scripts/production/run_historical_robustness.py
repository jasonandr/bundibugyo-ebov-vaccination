"""Historical-outbreak robustness analysis with the production C++ engine.

Two historical settings with constant transmission intensity (Rt held at the
historical maximum from prior EpiNow2 estimation), four intervention
strategies, matched simulation seeds across strategies within a setting.

  2007 Bundibugyo (Uganda): constant Rt = 2.37
  2012 Isiro (DRC):         constant Rt = 1.33

Strategies:
  base_ops               reporting 0.30 flat, tracing 0.30 flat,
                         detection_delay=4, tracing_delay=2, no vaccine
  enhanced_ops           reporting ramp 0.30->0.70 over 15 d, tracing ramp
                         0.30->0.80 over 15 d, detection_delay=2,
                         tracing_delay=1, no vaccine
  enhanced_plus_ring2    enhanced ops + ring-2 vaccination (VE 0.45)
  enhanced_plus_comm40   enhanced ops + 40% community vaccination
                         (trigger=1, delay=0, 14-day rollout, VE 0.45)

Comparators: enhanced_ops vs base_ops; enhanced_plus_ring2 vs enhanced_ops;
enhanced_plus_comm40 vs base_ops. Reductions are computed per matched
replicate pair (same seed), then summarized as medians with empirical
2.5th-97.5th percentiles.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REPO = ROOT.parent.parent

from ebola_stochastic_ring import generate_network, generate_network_clustered, simulate_ring_vaccination

N = 10_000
HORIZON_DAYS = 90
ARRAY_LEN = HORIZON_DAYS + 1
CHUNK_REPS = 25

SETTINGS = {
    "bundibugyo_2007": {"rt_max": 2.37, "network_seed": 202607231, "sim_seed0": 2026081000},
    "isiro_2012": {"rt_max": 1.33, "network_seed": 202607232, "sim_seed0": 2026091000},
}

COMPARATORS = {
    "base_ops": None,
    "enhanced_ops": "base_ops",
    "enhanced_plus_ring2": "enhanced_ops",
    "enhanced_plus_comm40": "base_ops",
}


def ramp(start, target, days=15):
    return np.linspace(start, target, days).tolist() + [target] * (ARRAY_LEN - days)


def strategy_kwargs(name):
    """Intervention overlay on top of the shared base parameters."""
    enhanced = {
        "reporting_rate": ramp(0.30, 0.70),
        "tracing_coverage": ramp(0.30, 0.80),
        "detection_delay": 2.0,
        "tracing_delay": 1.0,
    }
    if name == "base_ops":
        return {
            "reporting_rate": [0.30] * ARRAY_LEN,
            "tracing_coverage": [0.30] * ARRAY_LEN,
            "detection_delay": 4.0,
            "tracing_delay": 2.0,
            "max_vaccines": 0,
        }
    if name == "enhanced_ops":
        return {**enhanced, "max_vaccines": 0}
    if name == "enhanced_plus_ring2":
        return {**enhanced, "max_vaccines": None}
    if name == "enhanced_plus_comm40":
        return {
            **enhanced,
            "max_vaccines": 0,
            "community_vax_coverage": 0.40,
            "community_vax_trigger": 1,
            "community_vax_delay": 0.0,
            "community_vax_rollout_days": 14.0,
        }
    raise ValueError(name)


WORKER_GRAPHS = {}
WORKER_CFR = None
ACCEPTANCE = 0.90


def init_worker(network_seeds, base_cfr, topology="original", acceptance=0.90):
    global ACCEPTANCE
    ACCEPTANCE = acceptance
    global WORKER_CFR
    WORKER_CFR = base_cfr
    for setting, seed in network_seeds.items():
        if topology == "clustered":
            WORKER_GRAPHS[setting] = generate_network_clustered(N=N, seed=seed)
        else:
            np.random.seed(seed)
            WORKER_GRAPHS[setting] = generate_network(
                N=N, household_mean=5.2, community_mean=30.0, community_variance=160.0
            )


def run_chunk(task_spec):
    setting, strategy, rep_start, rep_end = task_spec
    cfg = SETTINGS[setting]
    graph = WORKER_GRAPHS[setting]
    rt_array = [cfg["rt_max"]] * ARRAY_LEN
    rows = []
    for replicate in range(rep_start, rep_end):
        seed = cfg["sim_seed0"] + replicate
        cases_frac, deaths_frac, doses, _ = simulate_ring_vaccination(
            graph,
            rt_array=rt_array,
            incubation_period=8.5,
            infectious_period=6.0,
            incubation_shape=2.0,
            infectious_shape=2.0,
            ring_radius=2,
            efficacy=0.45,
            sigmoidal_k=0.5,
            sigmoidal_d0=10.0,
            uptake=0.8,
            vaccine_acceptability=ACCEPTANCE,
            max_daily_traces=100,
            base_CFR=WORKER_CFR,
            initial_infected=15,
            initial_exposed=15,
            max_sim_time=HORIZON_DAYS,
            allow_pep=True,
            seed=seed,
            engine="cpp",
            **strategy_kwargs(strategy),
        )
        rows.append({
            "setting": setting,
            "strategy": strategy,
            "replicate": replicate,
            "seed": seed,
            "cases": int(round(cases_frac * N)),
            "deaths": int(round(deaths_frac * N)),
            "doses": int(doses),
        })
    return rows


def summarize(raw_rows, replicates):
    summaries = []
    for setting in SETTINGS:
        for strategy, comparator in COMPARATORS.items():
            strat = {(r["replicate"]): r for r in raw_rows
                     if r["setting"] == setting and r["strategy"] == strategy}
            row = {
                "setting": setting,
                "strategy": strategy,
                "comparator": comparator or "",
                "median_cases": float(np.median([r["cases"] for r in strat.values()])),
                "median_deaths": float(np.median([r["deaths"] for r in strat.values()])),
                "median_doses": float(np.median([r["doses"] for r in strat.values()])),
            }
            for metric in ("infection", "mortality"):
                row[f"{metric}_reduction_median"] = ""
                row[f"{metric}_reduction_p025"] = ""
                row[f"{metric}_reduction_p975"] = ""
            row["n_pairs"] = ""
            row["n_excluded_zero_comparator"] = ""
            if comparator:
                comp = {r["replicate"]: r for r in raw_rows
                        if r["setting"] == setting and r["strategy"] == comparator}
                inf_red, mort_red = [], []
                excluded = 0
                for rep in range(replicates):
                    c, s = comp[rep], strat[rep]
                    if c["cases"] <= 0 or c["deaths"] <= 0:
                        excluded += 1
                        continue
                    inf_red.append(100.0 * (c["cases"] - s["cases"]) / c["cases"])
                    mort_red.append(100.0 * (c["deaths"] - s["deaths"]) / c["deaths"])
                for metric, values in (("infection", inf_red), ("mortality", mort_red)):
                    arr = np.array(values)
                    row[f"{metric}_reduction_median"] = float(np.median(arr))
                    row[f"{metric}_reduction_p025"] = float(np.percentile(arr, 2.5))
                    row[f"{metric}_reduction_p975"] = float(np.percentile(arr, 97.5))
                row["n_pairs"] = len(inf_red)
                row["n_excluded_zero_comparator"] = excluded
            summaries.append(row)
    return summaries


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=REPO / "data_and_results/outputs/historical_robustness_production_20260723")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=625)
    parser.add_argument("--topology", choices=["original", "clustered"], default="original")
    parser.add_argument("--acceptance", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output}")
    args.output.mkdir(parents=True)

    base_cfr = float(json.loads((REPO / "data_and_results/fitted_parameters.json").read_text())["base_CFR"])

    specs = [
        (setting, strategy, rep_start, min(rep_start + CHUNK_REPS, args.replicates))
        for setting in SETTINGS
        for strategy in COMPARATORS
        for rep_start in range(0, args.replicates, CHUNK_REPS)
    ]
    network_seeds = {name: cfg["network_seed"] for name, cfg in SETTINGS.items()}
    raw_rows = []
    with mp.Pool(args.workers, initializer=init_worker,
                 initargs=(network_seeds, base_cfr, args.topology, args.acceptance)) as pool:
        for rows in pool.imap_unordered(run_chunk, specs):
            raw_rows.extend(rows)
    raw_rows.sort(key=lambda r: (r["setting"], r["strategy"], r["replicate"]))

    raw_path = args.output / "historical_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summaries = summarize(raw_rows, args.replicates)
    summary_path = args.output / "historical_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_head = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "code_sha256": {
            "run_historical_robustness.py": sha256(Path(__file__).resolve()),
            "ebola_stochastic_ring.py": sha256(ROOT / "ebola_stochastic_ring.py"),
            "ebola_stochastic_ring_cpp.cpp": sha256(ROOT / "ebola_stochastic_ring_cpp.cpp"),
        },
        "engine": "cpp (pooled daily onset-cohort allocator, use_cohort default)",
        "network": {
            "generator": ("generate_network_clustered (nested households in local community clusters + NB stubs)"
                      if args.topology == "clustered" else
                      "generate_network (two-layer DHS household + negative-binomial community)"),
            "topology": args.topology,
            "N": N,
            "household_mean": 5.2,
            "community_mean": 30.0,
            "community_variance": 160.0,
            "network_seeds": network_seeds,
            "note": "one network per setting, generated per worker with the fixed seed and reused across all strategies/replicates",
        },
        "settings": {
            name: {"constant_rt": cfg["rt_max"], "sim_seed_base": cfg["sim_seed0"],
                   "sim_seeds": f"{cfg['sim_seed0']}+i for i=0..{args.replicates - 1}"}
            for name, cfg in SETTINGS.items()
        },
        "replicates_per_strategy": args.replicates,
        "total_simulations": len(SETTINGS) * len(COMPARATORS) * args.replicates,
        "base_parameters": {
            "baseline_tau": 0.25,
            "incubation_period": 8.5, "infectious_period": 6.0,
            "incubation_shape": 2.0, "infectious_shape": 2.0,
            "base_CFR": base_cfr, "efficacy": 0.45,
            "sigmoidal_k": 0.5, "sigmoidal_d0": 10.0,
            "initial_infected": 15, "initial_exposed": 15,
            "max_sim_time": HORIZON_DAYS, "max_daily_traces": 100,
            "uptake": 0.8, "vaccine_acceptability": ACCEPTANCE, "allow_pep": True,
            "ring_radius": 2,
        },
        "strategies": {
            name: {k: (v if not isinstance(v, list) else f"array len {len(v)}: {v[0]:.4f}...{v[-1]:.4f}")
                   for k, v in strategy_kwargs(name).items()}
            for name in COMPARATORS
        },
        "comparators": COMPARATORS,
        "reductions": "per matched replicate pair (same seed); median and empirical 2.5/97.5 percentiles across replicates; pairs with zero comparator cases/deaths excluded",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
