import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ebola_stochastic_ring import calibrate_tau, generate_network, simulate_ring_vaccination
from paths import DATA_DIR, result_path


BASE_SEED = 20260630
POPULATION_SIZE = int(os.environ.get("FINAL_ESTIMATE_N", "5000"))
N_REPLICATES = int(os.environ.get("FINAL_ESTIMATE_REPS", "5000"))
N_WORKERS = int(os.environ.get("FINAL_ESTIMATE_WORKERS", "8"))
BASE_VACCINE_EFFECT = float(os.environ.get("FINAL_ESTIMATE_VACCINE_EFFECT", "0.45"))
SIGMOIDAL_K = 0.5
PARAMETER_PATH = result_path("fitted_parameters.json")
TRANSMISSION_MODE = os.environ.get("FINAL_ESTIMATE_TRANSMISSION", "rt").lower()
HOUSEHOLD_MEAN = float(os.environ.get("NETWORK_HOUSEHOLD_MEAN", "5.0"))
COMMUNITY_MEAN = float(os.environ.get("NETWORK_COMMUNITY_MEAN", "5.0"))
COMMUNITY_VARIANCE = float(os.environ.get("NETWORK_COMMUNITY_VARIANCE", "25.0"))
TAU_CALIBRATION_TRIALS = int(os.environ.get("TAU_CALIBRATION_TRIALS", "100"))
TAU_SCALE_DENOMINATOR = float(os.environ.get("FINAL_ESTIMATE_TAU_SCALE_DENOMINATOR", "2.5"))

ARRAY_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
ARRAY_COUNT = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))


def load_parameters():
    try:
        with open(PARAMETER_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_cfr_parameters():
    params = load_parameters()
    base_cfr = float(params.get("base_CFR", params.get("latest_adjusted_cfr", 0.454)))
    return base_cfr, base_cfr * 0.5


BASE_CFR, VAX_CFR = load_cfr_parameters()
PARAMS = load_parameters()
RT_ARRAY = PARAMS.get("Rt_array") if TRANSMISSION_MODE == "rt" else None
RT_MAX = max(RT_ARRAY) if RT_ARRAY else 1.66

def load_tau_array():
    if TRANSMISSION_MODE != "rt":
        return None
    cache_path = result_path("rt_calibrated_tau_array.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)["tau_array"]
    return None

TAU_ARRAY = load_tau_array()
if TAU_ARRAY is not None:
    RT_ARRAY = TAU_ARRAY
    BASELINE_TAU = 0.25
else:
    BASELINE_TAU = 0.25


def scaled_initial_count(name, default):
    fitted_population = float(PARAMS.get("N", POPULATION_SIZE))
    count = float(PARAMS.get(name, default)) * (POPULATION_SIZE / fitted_population)
    return max(1, int(round(count))) if count > 0 else 0


def scenario_definitions():
    scenarios = []
    
    def make_ramp(target, duration=15, max_time=91):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)
    
    base_reporting = [0.3] * 91
    base_tracing = [0.3] * 91
    enh_reporting = make_ramp(0.7)
    enh_tracing = make_ramp(0.8)

    def add_scenario(name, level, radius, vaccine_effect=0.45, reporting_arr=None, tracing_arr=None, delay=4.0, uptake=1.0, vaccine_acceptability=1.0, sig_d0=10.0, max_time=90, allow_pep=True, k=0.5, max_vaccines=None, comm_cov=0.0, comm_trig=0, comm_del=-1.0, comm_rollout=0.0, rt_override=None, infection_ve_override=None, vax_cfr_override=None):
        if reporting_arr is None: reporting_arr = enh_reporting
        if tracing_arr is None: tracing_arr = enh_tracing
        if vaccine_effect <= 0.0 and max_vaccines is None:
            max_vaccines = 0
            
        scenarios.append({
            "scenario": name,
            "level": str(level),
            "radius": radius,
            "vaccine_effect": vaccine_effect,
            "reporting_rate": reporting_arr,
            "tracing_coverage": tracing_arr,
            "detection_delay": delay,
            "uptake": uptake,
            "vaccine_acceptability": vaccine_acceptability,
            "sigmoidal_d0": sig_d0,
            "sigmoidal_k": k,
            "max_sim_time": max_time,
            "allow_pep": allow_pep,
            "max_vaccines": max_vaccines,
            "community_vax_coverage": comm_cov,
            "community_vax_trigger": comm_trig,
            "community_vax_delay": comm_del,
            "community_vax_rollout_days": comm_rollout,
            "rt_override": rt_override,
            "infection_ve_override": infection_ve_override,
            "vax_cfr_override": vax_cfr_override
        })

    # 1. Community-wide Vaccination (at outbreak declaration) - BASE OPS
    for cov in [0.2, 0.4, 0.6, 0.8]:
        add_scenario("analysis_10_comm_vax_base_ops", f"comm_vax_{int(cov*100)}", radius=1, vaccine_effect=BASE_VACCINE_EFFECT, reporting_arr=base_reporting, tracing_arr=base_tracing, max_vaccines=0, comm_cov=cov, comm_trig=1, comm_del=0.0, comm_rollout=14.0)
    
    # 2. Hybrid Strategy - BASE OPS
    add_scenario("analysis_11_hybrid_base_ops", "hybrid_40", radius=1, vaccine_effect=BASE_VACCINE_EFFECT, reporting_arr=base_reporting, tracing_arr=base_tracing, comm_cov=0.4, comm_trig=1, comm_del=0.0, comm_rollout=14.0)

    # 3. Timing Sensitivity - BASE OPS
    # Community Vax at 50% coverage, starting at declaration plus delay.
    for comm_delay in [0.0, 7.0, 14.0]:
        add_scenario("analysis_12_timing_base_ops", f"reactive_detect_delay_{int(comm_delay)}", radius=1, vaccine_effect=BASE_VACCINE_EFFECT, reporting_arr=base_reporting, tracing_arr=base_tracing, max_vaccines=0, comm_cov=0.5, comm_trig=1, comm_del=comm_delay, comm_rollout=14.0)

    # Baseline for Comparison
    add_scenario("analysis_1_reactive_ring", "no_vax_base_ops", radius=1, vaccine_effect=0.0, reporting_arr=base_reporting, tracing_arr=base_tracing)

    return scenarios


print("Generating global network topology...", flush=True)
np.random.seed(BASE_SEED)
GLOBAL_GRAPH = generate_network(
    POPULATION_SIZE,
    household_mean=HOUSEHOLD_MEAN,
    community_mean=COMMUNITY_MEAN,
    community_variance=COMMUNITY_VARIANCE,
)

def run_one(args):
    scenario, replicate, seed = args
    np.random.seed(seed)
    graph = GLOBAL_GRAPH
    kwargs = {}
    if scenario.get("allow_pep") is not None:
        kwargs.update({"allow_pep": scenario.get("allow_pep")})
    if scenario.get("sigmoidal_d0") is not None:
        kwargs.update({"sigmoidal_k": scenario.get("sigmoidal_k", SIGMOIDAL_K), "sigmoidal_d0": scenario.get("sigmoidal_d0")})

    current_vaccine_effect = float(scenario.get("vaccine_effect", BASE_VACCINE_EFFECT))
    if scenario.get("infection_ve_override") is not None:
        current_vaccine_effect = float(scenario["infection_ve_override"])
    
    current_vax_cfr = BASE_CFR * (1.0 - current_vaccine_effect)
    if scenario.get("vax_cfr_override") is not None:
        current_vax_cfr = float(scenario["vax_cfr_override"])

    if scenario["scenario"] == "vaccine_efficacy_scaled_cfr":
        current_vaccine_effect = float(scenario["efficacy"])
        current_vax_cfr = BASE_CFR * (1.0 - current_vaccine_effect)

    rt_array_to_use = RT_ARRAY
    if scenario.get("rt_override") is not None:
        rt_array_to_use = [float(scenario["rt_override"])] * 91

    cases, deaths, vaccines = simulate_ring_vaccination(
        graph,
        rt_array=rt_array_to_use,
        baseline_tau=BASELINE_TAU,
        incubation_period=8.5,
        infectious_period=6.0,
        uptake=scenario.get("uptake", 0.8),
        vaccine_effect=current_vaccine_effect,
        reporting_rate=scenario.get("reporting_rate", 0.7),
        tracing_coverage=scenario.get("tracing_coverage", -1.0),
        vaccine_acceptability=scenario.get("vaccine_acceptability", 1.0),
        detection_delay=scenario.get("detection_delay", 4.0),
        ring_radius=scenario["radius"],
        max_daily_traces=1000,
        max_vaccines=scenario.get("max_vaccines", None),
        base_CFR=BASE_CFR,
        vax_CFR=current_vax_cfr,
        initial_infected=scaled_initial_count("I0", 5),
        initial_exposed=scaled_initial_count("E0", 0),
        max_sim_time=scenario.get("max_sim_time", 90),
        engine='cpp',
        seed=seed,
        community_vax_coverage=scenario.get("community_vax_coverage", 0.0),
        community_vax_trigger=scenario.get("community_vax_trigger", 0),
        community_vax_delay=scenario.get("community_vax_delay", -1.0),
        community_vax_rollout_days=scenario.get("community_vax_rollout_days", 0.0),
        **kwargs,
    )

    return {
        "scenario": scenario["scenario"],
        "level": scenario["level"],
        "radius": scenario["radius"],
        "replicate": replicate,
        "seed": seed,
        "population_size": POPULATION_SIZE,
        "detection": scenario.get("reporting_rate", [0.7])[-1] if isinstance(scenario.get("reporting_rate"), list) else scenario.get("reporting_rate", 0.7),
        "tracing": scenario.get("tracing_coverage", [0.8])[-1] if isinstance(scenario.get("tracing_coverage"), list) else scenario.get("tracing_coverage", 0.8),
        "detection_delay": scenario.get("detection_delay", 4.0),
        "uptake": scenario.get("uptake", 0.8),
        "vaccine_acceptability": scenario.get("vaccine_acceptability", 1.0),
        "max_vaccines": -1 if scenario.get("max_vaccines", None) is None else scenario.get("max_vaccines"),
        "community_vax_coverage": scenario.get("community_vax_coverage", 0.0),
        "community_vax_trigger": scenario.get("community_vax_trigger", 0),
        "community_vax_delay": scenario.get("community_vax_delay", -1.0),
        "community_vax_rollout_days": scenario.get("community_vax_rollout_days", 0.0),
        "sigmoidal_d0": scenario.get("sigmoidal_d0", 10.0),
        "max_sim_time": scenario.get("max_sim_time", 90),
        "base_cfr": BASE_CFR,
        "vax_cfr": current_vax_cfr,
        "actual_vaccine_effect": current_vaccine_effect,
        "actual_vax_cfr": current_vax_cfr,
        "transmission_mode": TRANSMISSION_MODE,
        "baseline_tau": BASELINE_TAU,
        "rt_max": RT_MAX,
        "household_mean": HOUSEHOLD_MEAN,
        "community_mean": COMMUNITY_MEAN,
        "community_variance": COMMUNITY_VARIANCE,
        "cases_percent": cases * 100.0,
        "deaths_percent": deaths * 100.0,
        "cases_abs": cases * POPULATION_SIZE,
        "deaths_abs": deaths * POPULATION_SIZE,
        "vaccines": vaccines,
        "vaccines_percent": vaccines / POPULATION_SIZE * 100.0,
        "rt_override": scenario.get("rt_override", -1.0),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


RAW_FIELDS = [
    "scenario", "level", "radius", "replicate", "seed", "population_size", "detection",
    "tracing", "detection_delay", "uptake", "vaccine_acceptability", "max_vaccines",
    "community_vax_coverage", "community_vax_trigger", "community_vax_delay", "community_vax_rollout_days",
    "max_sim_time", "base_cfr", "vax_cfr", "transmission_mode", "baseline_tau", "rt_max",
    "household_mean", "community_mean", "community_variance",
    "cases_percent", "deaths_percent", "cases_abs", "deaths_abs", "vaccines", "vaccines_percent",
    "actual_vaccine_effect", "actual_vax_cfr", "sigmoidal_d0", "rt_override"
]


def load_existing_rows(path):
    if not path.exists():
        return []

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in RAW_FIELDS):
            return []
        for row in reader:
            
            rows.append({
                "scenario": row["scenario"],
                "level": row["level"],
                "radius": int(row["radius"]),
                "replicate": int(row["replicate"]),
                "seed": int(row["seed"]),
                "population_size": int(row["population_size"]),
                "detection": float(row.get("detection") or 0.7),
                "tracing": float(row.get("tracing") or 0.8),
                "detection_delay": float(row.get("detection_delay") or 4.0),
                "uptake": float(row.get("uptake") or 0.8),
                "vaccine_acceptability": float(row.get("vaccine_acceptability") or 1.0),
                "max_vaccines": int(float(row.get("max_vaccines") or -1)),
                "community_vax_coverage": float(row.get("community_vax_coverage") or 0.0),
                "community_vax_trigger": int(float(row.get("community_vax_trigger") or 0)),
                "community_vax_delay": float(row.get("community_vax_delay") or -1.0),
                "community_vax_rollout_days": float(row.get("community_vax_rollout_days") or 0.0),
                "max_sim_time": int(float(row.get("max_sim_time") or 90)),
                "base_cfr": float(row["base_cfr"]),
                "vax_cfr": float(row["vax_cfr"]),
                "transmission_mode": row["transmission_mode"],
                "baseline_tau": float(row["baseline_tau"]),
                "rt_max": float(row["rt_max"]),
                "household_mean": float(row["household_mean"]),
                "community_mean": float(row["community_mean"]),
                "community_variance": float(row["community_variance"]),
                "cases_percent": float(row["cases_percent"]),
                "deaths_percent": float(row["deaths_percent"]),
                "cases_abs": float(row.get("cases_abs", 0.0)),
                "deaths_abs": float(row.get("deaths_abs", 0.0)),
                "vaccines": int(float(row["vaccines"])),
                "vaccines_percent": float(row["vaccines_percent"]),
                "actual_vaccine_effect": float(row.get("actual_vaccine_effect") or 0.0),
                "actual_vax_cfr": float(row.get("actual_vax_cfr") or 0.0),
            })
    return rows


def main():
    scenarios = scenario_definitions()
    
    if ARRAY_ID != -1:
        Path("data_and_results/base_ops_chunks").mkdir(parents=True, exist_ok=True)
        raw_path = Path(f"data_and_results/base_ops_chunks/base_ops_results_{ARRAY_ID}.csv")
        summary_path = result_path(f"final_high_replicate_summary_{ARRAY_ID}.csv")
        summary_md_path = result_path(f"final_high_replicate_summary_{ARRAY_ID}.md")
        npz_path = DATA_DIR / f"final_high_replicate_estimates_{ARRAY_ID}.npz"
    else:
        raw_path = result_path("final_high_replicate_raw.csv")
        summary_path = result_path("final_high_replicate_summary.csv")
        summary_md_path = result_path("final_high_replicate_summary.md")
        npz_path = DATA_DIR / "final_high_replicate_estimates.npz"

    # Always load existing progress from the MAIN file to avoid re-running cached scenarios
    main_path = result_path("final_high_replicate_raw.csv")
    all_rows = load_existing_rows(main_path)
    for scenario in scenarios:
        label = f"{scenario['scenario']} level={scenario['level']} radius={scenario['radius']}"
        completed = [
            row for row in all_rows
            if row["scenario"] == scenario["scenario"]
            and row["level"] == scenario["level"]
            and row["radius"] == scenario["radius"]
        ]
        if len(completed) >= N_REPLICATES:
            print(f"Skipping {label}; {len(completed)} replicates already complete", flush=True)
            continue

        print(
            f"Running {label} with {N_REPLICATES - len(completed)} remaining "
            f"of {N_REPLICATES} replicates at N={POPULATION_SIZE}",
            flush=True,
        )
        args = []
        for replicate in range(len(completed), N_REPLICATES):
            if ARRAY_ID != -1:
                # e.g., ARRAY_ID from 1 to 100
                # (replicate % ARRAY_COUNT) + 1 == ARRAY_ID
                if (replicate % ARRAY_COUNT) + 1 != ARRAY_ID:
                    continue
            seed = BASE_SEED + replicate
            args.append((scenario, replicate, seed))
            
        if not args:
            continue
        if N_WORKERS <= 1:
            rows = []
            for idx, arg in enumerate(args, start=1):
                rows.append(run_one(arg))
                if idx % 500 == 0:
                    print(f"  completed {idx}/{len(args)}", flush=True)
        else:
            with Pool(processes=N_WORKERS) as pool:
                rows = []
                for idx, row in enumerate(pool.imap_unordered(run_one, args, chunksize=10), start=1):
                    rows.append(row)
                    if idx % 1000 == 0:
                        print(f"  completed {idx}/{len(args)}", flush=True)
        all_rows.extend(rows)
        write_csv(raw_path, all_rows, RAW_FIELDS)

    summary_rows = []
    grouped = {}
    for row in all_rows:
        key = (row["scenario"], row["level"], row["radius"])
        grouped.setdefault(key, []).append(row)

    for (scenario, level, radius), rows in grouped.items():
        cases = np.array([row["cases_percent"] for row in rows], dtype=float)
        deaths = np.array([row["deaths_percent"] for row in rows], dtype=float)
        vaccines = np.array([row["vaccines_percent"] for row in rows], dtype=float)
        detection = float(rows[0]["detection"])
        cases_summary = summarize(cases)
        deaths_summary = summarize(deaths)
        vaccine_summary = summarize(vaccines)
        summary_rows.append({
            "scenario": scenario,
            "level": level,
            "radius": radius,
            "n": len(rows),
            "population_size": POPULATION_SIZE,
            "detection": detection,
            "vaccine_effect": rows[0]["actual_vaccine_effect"],
            "vax_cfr": rows[0]["actual_vax_cfr"],
            "transmission_mode": rows[0]["transmission_mode"],
            "baseline_tau": rows[0]["baseline_tau"],
            "rt_max": rows[0]["rt_max"],
            "household_mean": rows[0]["household_mean"],
            "community_mean": rows[0]["community_mean"],
            "community_variance": rows[0]["community_variance"],
            "cases_percent_mean": cases_summary["mean"],
            "cases_percent_median": cases_summary["median"],
            "cases_percent_p25": cases_summary["p25"],
            "cases_percent_p75": cases_summary["p75"],
            "deaths_percent_mean": deaths_summary["mean"],
            "deaths_percent_median": deaths_summary["median"],
            "deaths_percent_p25": deaths_summary["p25"],
            "deaths_percent_p75": deaths_summary["p75"],
            "vaccines_percent_mean": vaccine_summary["mean"],
            "vaccines_percent_p25": vaccine_summary["p25"],
            "vaccines_percent_p75": vaccine_summary["p75"],
        })

    fields = list(summary_rows[0].keys())
    write_csv(summary_path, summary_rows, fields)
    with open(summary_md_path, "w") as f:
        f.write("# Final High-Replicate Estimates\n\n")
        f.write(
            f"Population size: {POPULATION_SIZE}. Replicates per scenario: {N_REPLICATES}. "
            f"Transmission mode: {TRANSMISSION_MODE}; calibrated baseline tau: {BASELINE_TAU:.4f}; "
            f"target peak Rt: {RT_MAX:.2f}. Network household mean: {HOUSEHOLD_MEAN:.1f}; "
            f"community degree mean: {COMMUNITY_MEAN:.1f}; community degree variance: {COMMUNITY_VARIANCE:.1f}. "
            f"Base vaccine effect: {BASE_VACCINE_EFFECT * 100:.1f}% against infection/PEP and death; "
            f"baseline CFR: {BASE_CFR * 100:.1f}%; base breakthrough CFR: {BASE_CFR * (1.0 - BASE_VACCINE_EFFECT) * 100:.1f}%.\n\n"
        )
        f.write("| Scenario | Level | Radius | Detection | Vaccine effect | n | Cases %, median (IQR) | Deaths %, median (IQR) | Vaccinated, mean % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {scenario} | {level} | {radius} | {detection:.1f} | {vaccine_effect:.2f} | {n} | "
                "{cases_percent_median:.2f} ({cases_percent_p25:.2f}-{cases_percent_p75:.2f}) | "
                "{deaths_percent_median:.2f} ({deaths_percent_p25:.2f}-{deaths_percent_p75:.2f}) | "
                "{vaccines_percent_mean:.2f} |\n".format(**row)
            )

    np.savez_compressed(
        npz_path,
        scenario=np.array([row["scenario"] for row in all_rows]),
        level=np.array([row["level"] for row in all_rows]),
        radius=np.array([row["radius"] for row in all_rows]),
        detection=np.array([row["detection"] for row in all_rows], dtype=float),
        base_cfr=np.array([row["base_cfr"] for row in all_rows], dtype=float),
        vax_cfr=np.array([row["vax_cfr"] for row in all_rows], dtype=float),
        actual_vaccine_effect=np.array([row["actual_vaccine_effect"] for row in all_rows], dtype=float),
        actual_vax_cfr=np.array([row["actual_vax_cfr"] for row in all_rows], dtype=float),
        baseline_tau=np.array([row["baseline_tau"] for row in all_rows], dtype=float),
        rt_max=np.array([row["rt_max"] for row in all_rows], dtype=float),
        community_mean=np.array([row["community_mean"] for row in all_rows], dtype=float),
        community_variance=np.array([row["community_variance"] for row in all_rows], dtype=float),
        cases_percent=np.array([row["cases_percent"] for row in all_rows], dtype=float),
        deaths_percent=np.array([row["deaths_percent"] for row in all_rows], dtype=float),
        vaccines_percent=np.array([row["vaccines_percent"] for row in all_rows], dtype=float),
    )

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
