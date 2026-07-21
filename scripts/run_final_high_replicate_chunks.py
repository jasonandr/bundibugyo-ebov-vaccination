import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from run_final_high_replicate_estimates import (
    BASELINE_TAU,
    BASE_CFR,
    COMMUNITY_MEAN,
    COMMUNITY_VARIANCE,
    DATA_DIR,
    HOUSEHOLD_MEAN,
    N_REPLICATES,
    POPULATION_SIZE,
    RAW_FIELDS,
    RT_MAX,
    TRANSMISSION_MODE,
    VAX_CFR,
    load_existing_rows,
    result_path,
    scenario_definitions,
    summarize,
    write_csv,
)


N_WORKERS = int(os.environ.get("FINAL_CHUNK_WORKERS", "8"))
CHUNK_SIZE = int(os.environ.get("FINAL_CHUNK_SIZE", "1000"))
CHUNK_DIR = result_path("final_high_replicate_chunks")


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "scenario": row["scenario"],
                "level": row["level"],
                "radius": int(row["radius"]),
                "replicate": int(row["replicate"]),
                "seed": int(row["seed"]),
                "population_size": int(row["population_size"]),
                "detection": float(row["detection"]),
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
                "vaccines": int(float(row["vaccines"])),
                "vaccines_percent": float(row["vaccines_percent"]),
            })
    return rows


def run_commands(commands):
    running = []
    completed = 0
    total = len(commands)
    while commands or running:
        while commands and len(running) < N_WORKERS:
            cmd = commands.pop(0)
            running.append(subprocess.Popen(cmd))
        next_running = []
        for proc in running:
            rc = proc.poll()
            if rc is None:
                next_running.append(proc)
            elif rc == 0:
                completed += 1
                print(f"  completed chunk {completed}/{total}")
                print(f"Finished chunk: {completed}/{total}")
            else:
                raise RuntimeError(f"Chunk command failed with exit code {rc}")
        running = next_running
        if running:
            time.sleep(1)


def build_summary(all_rows):
    baselines = {}
    for detection in [0.4, 0.6, 0.7, 0.8]:
        rows = [
            row for row in all_rows
            if row["scenario"] == "no_vaccination" and row["detection"] == detection
        ]
        rows.sort(key=lambda x: x["replicate"])
        baseline_cases = np.array([row["cases_percent"] for row in rows], dtype=float)
        baseline_deaths = np.array([row["deaths_percent"] for row in rows], dtype=float)
        baselines[detection] = {
            "cases_mean": float(np.mean(baseline_cases)),
            "deaths_mean": float(np.mean(baseline_deaths)),
            "cases_array": baseline_cases,
            "deaths_array": baseline_deaths,
        }

    summary_rows = []
    grouped = {}
    for row in all_rows:
        key = (row["scenario"], row["level"], row["radius"])
        grouped.setdefault(key, []).append(row)

    for (scenario, level, radius), rows in grouped.items():
        rows.sort(key=lambda x: x["replicate"])
        cases = np.array([row["cases_percent"] for row in rows], dtype=float)
        deaths = np.array([row["deaths_percent"] for row in rows], dtype=float)
        vaccines = np.array([row["vaccines_percent"] for row in rows], dtype=float)
        detection = float(rows[0]["detection"])
        baseline_cases_mean = baselines[detection]["cases_mean"]
        baseline_deaths_mean = baselines[detection]["deaths_mean"]
        
        base_c = baselines[detection]["cases_array"]
        base_d = baselines[detection]["deaths_array"]
        cases_averted = np.where(base_c > 0, (base_c - cases) / base_c * 100.0, 0.0)
        deaths_averted = np.where(base_d > 0, (base_d - deaths) / base_d * 100.0, 0.0)
        cases_summary = summarize(cases_averted)
        deaths_summary = summarize(deaths_averted)
        vaccine_summary = summarize(vaccines)
        summary_rows.append({
            "scenario": scenario,
            "level": level,
            "radius": radius,
            "n": len(rows),
            "population_size": POPULATION_SIZE,
            "detection": detection,
            "transmission_mode": rows[0]["transmission_mode"],
            "baseline_tau": rows[0]["baseline_tau"],
            "rt_max": rows[0]["rt_max"],
            "household_mean": rows[0]["household_mean"],
            "community_mean": rows[0]["community_mean"],
            "community_variance": rows[0]["community_variance"],
            "baseline_cases_percent_mean": baseline_cases_mean,
            "baseline_deaths_percent_mean": baseline_deaths_mean,
            "cases_averted_mean": cases_summary["mean"],
            "cases_averted_median": cases_summary["median"],
            "cases_averted_p2_5": cases_summary["p2_5"],
            "cases_averted_p97_5": cases_summary["p97_5"],
            "deaths_averted_mean": deaths_summary["mean"],
            "deaths_averted_median": deaths_summary["median"],
            "deaths_averted_p2_5": deaths_summary["p2_5"],
            "deaths_averted_p97_5": deaths_summary["p97_5"],
            "vaccines_percent_mean": vaccine_summary["mean"],
            "vaccines_percent_p2_5": vaccine_summary["p2_5"],
            "vaccines_percent_p97_5": vaccine_summary["p97_5"],
        })
    return sorted(summary_rows, key=lambda r: (r["scenario"], str(r["level"]), r["radius"])), baselines


def write_outputs(all_rows, summary_rows, baselines):
    raw_path = result_path("final_high_replicate_raw.csv")
    summary_path = result_path("final_high_replicate_summary.csv")
    summary_md_path = result_path("final_high_replicate_summary.md")
    npz_path = DATA_DIR / "final_high_replicate_estimates.npz"

    write_csv(raw_path, all_rows, RAW_FIELDS)
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    with open(summary_md_path, "w") as f:
        f.write("# Final High-Replicate Estimates\n\n")
        f.write(
            f"Population size: {POPULATION_SIZE}. Replicates per scenario: {N_REPLICATES}. "
            f"Transmission mode: {TRANSMISSION_MODE}; calibrated baseline tau: {BASELINE_TAU:.4f}; "
            f"target peak Rt: {RT_MAX:.2f}. Network household mean: {HOUSEHOLD_MEAN:.1f}; "
            f"community degree mean: {COMMUNITY_MEAN:.1f}; community degree variance: {COMMUNITY_VARIANCE:.1f}. "
            f"Baseline CFR: {BASE_CFR * 100:.1f}%; post-exposure CFR floor: {VAX_CFR * 100:.1f}%. "
            "Percent averted values are calculated relative to an explicit no-vaccination "
            "comparator with the same case detection and isolation assumptions.\n\n"
        )
        f.write("| Detection | No-vaccination attack %, mean | No-vaccination mortality %, mean |\n")
        f.write("|---:|---:|---:|\n")
        for detection, baseline in baselines.items():
            f.write(f"| {detection:.1f} | {baseline['cases_mean']:.2f} | {baseline['deaths_mean']:.2f} |\n")
        f.write("\n")
        f.write("| Scenario | Level | Radius | Detection | n | Cases averted, median (95% UI) | Deaths averted, median (95% UI) | Vaccinated, mean % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {scenario} | {level} | {radius} | {detection:.1f} | {n} | "
                "{cases_averted_median:.1f} ({cases_averted_p2_5:.1f}-{cases_averted_p97_5:.1f}) | "
                "{deaths_averted_median:.1f} ({deaths_averted_p2_5:.1f}-{deaths_averted_p97_5:.1f}) | "
                "{vaccines_percent_mean:.1f} |\n".format(**row)
            )

    np.savez_compressed(
        npz_path,
        scenario=np.array([row["scenario"] for row in all_rows]),
        level=np.array([row["level"] for row in all_rows]),
        radius=np.array([row["radius"] for row in all_rows]),
        detection=np.array([row["detection"] for row in all_rows], dtype=float),
        base_cfr=np.array([row["base_cfr"] for row in all_rows], dtype=float),
        vax_cfr=np.array([row["vax_cfr"] for row in all_rows], dtype=float),
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


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = result_path("final_high_replicate_raw.csv")
    all_rows = load_existing_rows(raw_path)
    scenarios = scenario_definitions()
    for scenario_index, scenario in enumerate(scenarios):
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

        print(f"Running {label}; {N_REPLICATES - len(completed)} missing replicates", flush=True)
        completed_reps = {row["replicate"] for row in completed}
        commands = []
        chunk_paths = []
        start = 0
        while start < N_REPLICATES:
            end = min(start + CHUNK_SIZE, N_REPLICATES)
            reps = [rep for rep in range(start, end) if rep not in completed_reps]
            if reps:
                output = CHUNK_DIR / f"scenario_{scenario_index:02d}_{start:05d}_{end:05d}.csv"
                chunk_paths.append(output)
                if not output.exists():
                    commands.append([
                        sys.executable,
                        str(Path(__file__).with_name("run_final_replicate_chunk.py")),
                        "--scenario-index", str(scenario_index),
                        "--start", str(start),
                        "--end", str(end),
                        "--output", str(output),
                    ])
            start = end
        run_commands(commands)
        for path in chunk_paths:
            all_rows.extend(read_rows(path))
        write_csv(raw_path, all_rows, RAW_FIELDS)

    summary_rows, baselines = build_summary(all_rows)
    write_outputs(all_rows, summary_rows, baselines)


if __name__ == "__main__":
    main()
