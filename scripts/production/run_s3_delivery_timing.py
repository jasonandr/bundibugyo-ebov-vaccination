"""Vaccination timing among eventual cases (delivery-timing analysis).

For two strategies (community vaccination at 40% under base operations;
Ring 2 vaccination under enhanced operations), classifies each vaccinated
eventual case by its state at the time of vaccination: before exposure
(susceptible), during incubation (exposed), or after onset (infectious or
later).  Uses the C++ engine's mechanism-level outputs with matched seeds.
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
HORIZON = 90
BASE_SEED = 2026131000
FIELDS = ["after_onset", "analysis", "before_exposure", "coverage", "during_incubation",
          "enhanced", "max_vaccines", "n_vaccinated_cases", "radius", "replicate",
          "seed", "strategy"]
WORKER_GRAPH = None
WORKER_PARAMS = None
ACCEPTANCE = 0.90


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ramp(target, duration=15):
    return np.linspace(0.30, target, duration).tolist() + [target] * (HORIZON + 1 - duration)


def scenario_kwargs(name):
    base = dict(ring_radius=2, efficacy=0.45, immune_delay=10.0, uptake=0.8,
                detection_delay=4.0, tracing_delay=2.0, max_daily_traces=100,
                sigmoidal_k=0.5, sigmoidal_d0=10.0, allow_pep=True,
                incubation_shape=2.0, infectious_shape=2.0,
                reporting_rate=[0.30] * (HORIZON + 1),
                tracing_coverage=[0.30] * (HORIZON + 1),
                vaccine_acceptability=ACCEPTANCE, max_vaccines=0)
    if name == "community40_base":
        return {**base, "community_vax_coverage": 0.4, "community_vax_trigger": 1,
                "community_vax_delay": 0.0, "community_vax_rollout_days": 14.0}
    if name == "ring2_enhanced":
        return {**base, "reporting_rate": ramp(0.70), "tracing_coverage": ramp(0.80),
                "detection_delay": 2.0, "tracing_delay": 1.0, "max_vaccines": None}
    raise ValueError(name)


def initialise_worker(params, network_cache_path, acceptance):
    global WORKER_GRAPH, WORKER_PARAMS, ACCEPTANCE
    from network_cache import load_cached_network
    WORKER_GRAPH = load_cached_network(network_cache_path)
    WORKER_PARAMS = params
    ACCEPTANCE = acceptance


def run_replicate(replicate_id):
    params = WORKER_PARAMS
    rt = list(params["Rt_array"])
    rt.extend([rt[-1]] * max(0, HORIZON + 1 - len(rt)))
    rows = []
    for name in ("community40_base", "ring2_enhanced"):
        seed = BASE_SEED + replicate_id
        res = simulate_ring_vaccination(
            WORKER_GRAPH, rt_array=rt, baseline_tau=0.25,
            incubation_period=8.5, infectious_period=6.0,
            base_CFR=float(params["base_CFR"]), initial_infected=15,
            initial_exposed=15, max_sim_time=HORIZON, seed=seed,
            engine="cpp", return_mechanism=True, **scenario_kwargs(name))
        state_at_vax = np.asarray(res["state_at_vaccination"], dtype=float)
        onset = np.asarray(res["onset_time"], dtype=float)
        vaccinated_cases = (state_at_vax != -1) & (onset != -1)
        states = state_at_vax[vaccinated_cases]
        before = int((states == 0).sum())
        during = int((states == 1).sum())
        after = int((states >= 2).sum())
        rows.append({
            "after_onset": after, "analysis": "supp_s3_delivery",
            "before_exposure": before,
            "coverage": 0.4 if name == "community40_base" else "",
            "during_incubation": during,
            "enhanced": "" if name == "community40_base" else 1,
            "max_vaccines": 0 if name == "community40_base" else "",
            "n_vaccinated_cases": int(vaccinated_cases.sum()),
            "radius": 2, "replicate": replicate_id, "seed": seed, "strategy": name})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--network-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceptance", type=float, default=0.90)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not args.network_cache.exists():
        raise FileNotFoundError(f"Network cache not found: {args.network_cache}")

    params = json.loads(FITTED_PATH.read_text())
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "supp_s3_delivery_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with mp.Pool(args.workers, initializer=initialise_worker,
                     initargs=(params, str(args.network_cache), args.acceptance)) as pool:
            for rows in pool.imap_unordered(run_replicate, range(args.replicates)):
                writer.writerows(rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicates": args.replicates,
        "seed_schedule": f"{BASE_SEED}+replicate, matched across strategies",
        "classification": "state at vaccination among vaccinated eventual cases: "
                          "0=susceptible (before exposure), 1=exposed (during incubation), "
                          ">=2 (after onset)",
        "network_cache": str(args.network_cache),
        "network_cache_sha256": sha256(args.network_cache),
        "vaccine_acceptability": args.acceptance,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {raw_path} and manifest.json")


if __name__ == "__main__":
    main()
