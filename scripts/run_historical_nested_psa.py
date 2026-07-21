import csv
import json
import os
import time
from multiprocessing import Pool
import numpy as np

from ebola_stochastic_ring import calibrate_tau, generate_network, simulate_ring_vaccination
from paths import DATA_DIR, result_path

BASE_SEED = 20261101
POPULATION_SIZE = 10000
N_DRAWS = 200
N_REPS = 100
N_WORKERS = 8

HOUSEHOLD_MEAN = 5.2
COMMUNITY_MEAN = 5.0
COMMUNITY_VARIANCE = 25.0

def make_ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)

def get_scenarios():
    base_reporting = [0.3] * 91
    base_tracing = [0.3] * 91
    enh_reporting = make_ramp(0.7)
    enh_tracing = make_ramp(0.8)

    return [
        {"name": "Base Operations", "reporting": base_reporting, "tracing": base_tracing, "vaccine_effect": 0.0, "radius": 1, "comm_cov": 0.0, "comm_trig": 0, "comm_del": -1.0, "comm_rollout": 0.0},
        {"name": "Enhanced Operations", "reporting": enh_reporting, "tracing": enh_tracing, "vaccine_effect": 0.0, "radius": 1, "comm_cov": 0.0, "comm_trig": 0, "comm_del": -1.0, "comm_rollout": 0.0},
        {"name": "Reactive Ring Vaccination", "reporting": enh_reporting, "tracing": enh_tracing, "vaccine_effect": 0.45, "radius": 1, "comm_cov": 0.0, "comm_trig": 0, "comm_del": -1.0, "comm_rollout": 0.0},
        {"name": "Community Vaccination", "reporting": enh_reporting, "tracing": enh_tracing, "vaccine_effect": 0.45, "radius": 1, "comm_cov": 0.60, "comm_trig": 1, "comm_del": 0.0, "comm_rollout": 14.0},
    ]

def load_cfr_parameters():
    with open(result_path("fitted_parameters.json"), "r") as f:
        params = json.load(f)
    base_cfr = float(params.get("base_CFR", params.get("latest_adjusted_cfr", 0.454)))
    return base_cfr

def run_one_draw(args):
    draw_idx, dataset_year = args
    
    np.random.seed(BASE_SEED + int(dataset_year) * 10000 + draw_idx)
    
    # 1. Parameter draws (Outer loop)
    psa_incubation = np.random.uniform(7.0, 10.0)
    psa_infectious = np.random.uniform(4.0, 8.0)
    psa_detection_delay = np.random.uniform(2.0, 6.0)
    psa_vaccine_effect = np.random.triangular(max(0.0, 0.45 - 0.25), 0.45, min(1.0, 0.45 + 0.25))
    psa_base_cfr = np.random.uniform(0.40, 0.75) 
    
    rep_target = np.random.uniform(0.5, 0.9) # target 0.7
    trc_target = np.random.uniform(0.5, 1.0) # target 0.8
    base_rep_target = np.random.uniform(0.2, 0.4) # target 0.3
    base_trc_target = np.random.uniform(0.2, 0.4) # target 0.3

    with open(f'/Users/jasonandrews/repos/ebola vaccination modeling/data_and_results/historical_params_{dataset_year}.json', 'r') as f:
        params = json.load(f)
    R_max = params.get('R_max', 2.0)

    # 2. Network & Calibration
    G = generate_network(POPULATION_SIZE, household_mean=HOUSEHOLD_MEAN, community_mean=COMMUNITY_MEAN, community_variance=COMMUNITY_VARIANCE)
    baseline_tau = calibrate_tau(G, R_max, 1.0/psa_infectious, num_trials=10)

    scenarios = get_scenarios()
    
    results = {}
    for sc in scenarios:
        results[sc["name"]] = {"cases": 0.0, "deaths": 0.0}

    # 3. Inner loop (Stochastic Replicates)
    for rep in range(N_REPS):
        seed = BASE_SEED + int(dataset_year) * 1000000 + draw_idx * 1000 + rep
        
        for sc in scenarios:
            rep_arr = sc["reporting"]
            trc_arr = sc["tracing"]
            
            # Apply PSA multipliers
            target_r = rep_arr[-1]
            if target_r == 0.7:
                rep_arr = [min(1.0, x * (rep_target / 0.7)) for x in rep_arr]
            elif target_r == 0.3:
                rep_arr = [min(1.0, x * (base_rep_target / 0.3)) for x in rep_arr]
                
            target_t = trc_arr[-1]
            if target_t == 0.8:
                trc_arr = [min(1.0, x * (trc_target / 0.8)) for x in trc_arr]
            elif target_t == 0.3:
                trc_arr = [min(1.0, x * (base_trc_target / 0.3)) for x in trc_arr]

            vax_eff = psa_vaccine_effect if sc["vaccine_effect"] > 0.0 else 0.0
            vax_cfr = psa_base_cfr * (1.0 - vax_eff)

            res = simulate_ring_vaccination(
                G,
                rt_array=None, # None = use constant baseline_tau * beta_modifiers
                baseline_tau=baseline_tau,
                incubation_period=psa_incubation,
                infectious_period=psa_infectious,
                uptake=0.8,
                vaccine_effect=vax_eff,
                reporting_rate=rep_arr,
                tracing_coverage=trc_arr,
                vaccine_acceptability=1.0,
                detection_delay=psa_detection_delay,
                ring_radius=sc["radius"],
                max_daily_traces=1000,
                max_vaccines=0 if vax_eff == 0.0 else None,
                base_CFR=psa_base_cfr,
                vax_CFR=vax_cfr,
                initial_infected=5,
                initial_exposed=0,
                max_sim_time=90,
                engine='cpp',
                seed=seed,
                community_vax_coverage=sc["comm_cov"],
                community_vax_trigger=sc["comm_trig"],
                community_vax_delay=sc["comm_del"],
                community_vax_rollout_days=sc["comm_rollout"],
            )
            cases, deaths = res[0], res[1]
            results[sc["name"]]["cases"] += cases * POPULATION_SIZE / N_REPS
            results[sc["name"]]["deaths"] += deaths * POPULATION_SIZE / N_REPS
            
    return {
        "draw": draw_idx,
        "dataset": dataset_year,
        "results": results
    }

def main():
    print(f"Starting Nested PSA: {N_DRAWS} draws, {N_REPS} reps per draw, N={POPULATION_SIZE}", flush=True)
    
    args = []
    for dataset in ["2007", "2012"]:
        for draw in range(N_DRAWS):
            args.append((draw, dataset))
            
    completed = []
    with Pool(processes=N_WORKERS) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_one_draw, args, chunksize=1)):
            completed.append(res)
            if (idx + 1) % 5 == 0:
                print(f"Completed {idx + 1}/{len(args)} draws", flush=True)
                
    # Process results
    metrics = {
        "2007": {"Enhanced Operations (vs Base)": {"cases": [], "deaths": []},
                 "Reactive Ring Vaccination (vs Enhanced)": {"cases": [], "deaths": []},
                 "Community Vaccination (vs Base)": {"cases": [], "deaths": []}},
        "2012": {"Enhanced Operations (vs Base)": {"cases": [], "deaths": []},
                 "Reactive Ring Vaccination (vs Enhanced)": {"cases": [], "deaths": []},
                 "Community Vaccination (vs Base)": {"cases": [], "deaths": []}}
    }
    
    for c in completed:
        ds = c["dataset"]
        r = c["results"]
        
        c_base = r["Base Operations"]["cases"]
        d_base = r["Base Operations"]["deaths"]
        
        c_enh = r["Enhanced Operations"]["cases"]
        d_enh = r["Enhanced Operations"]["deaths"]
        
        c_ring = r["Reactive Ring Vaccination"]["cases"]
        d_ring = r["Reactive Ring Vaccination"]["deaths"]
        
        c_comm = r["Community Vaccination"]["cases"]
        d_comm = r["Community Vaccination"]["deaths"]
        
        # Protect against division by zero if cases are exactly 0 (unlikely with 100 reps but possible)
        if c_base > 0:
            metrics[ds]["Enhanced Operations (vs Base)"]["cases"].append((c_base - c_enh) / c_base * 100)
            metrics[ds]["Community Vaccination (vs Base)"]["cases"].append((c_base - c_comm) / c_base * 100)
        
        if d_base > 0:
            metrics[ds]["Enhanced Operations (vs Base)"]["deaths"].append((d_base - d_enh) / d_base * 100)
            metrics[ds]["Community Vaccination (vs Base)"]["deaths"].append((d_base - d_comm) / d_base * 100)
            
        if c_enh > 0:
            metrics[ds]["Reactive Ring Vaccination (vs Enhanced)"]["cases"].append((c_enh - c_ring) / c_enh * 100)
            
        if d_enh > 0:
            metrics[ds]["Reactive Ring Vaccination (vs Enhanced)"]["deaths"].append((d_enh - d_ring) / d_enh * 100)
        
    out_rows = []
    for sc_name in ["Enhanced Operations (vs Base)", "Reactive Ring Vaccination (vs Enhanced)", "Community Vaccination (vs Base)"]:
        for ds in ["2007", "2012"]:
            c_arr = metrics[ds][sc_name]["cases"]
            d_arr = metrics[ds][sc_name]["deaths"]
            
            c_med = np.median(c_arr)
            c_lo = np.percentile(c_arr, 2.5)
            c_hi = np.percentile(c_arr, 97.5)
            
            d_med = np.median(d_arr)
            d_lo = np.percentile(d_arr, 2.5)
            d_hi = np.percentile(d_arr, 97.5)
            
            out_rows.append({
                "Scenario": sc_name,
                "Outbreak": ds,
                "Infection reduction % median (95% UI)": f"{c_med:.1f}% ({c_lo:.1f}-{c_hi:.1f}%)",
                "Mortality reduction % median (95% UI)": f"{d_med:.1f}% ({d_lo:.1f}-{d_hi:.1f}%)"
            })
            
    out_path = "data_and_results/historical_robustness_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
        
    print(f"Wrote updated table to {out_path}", flush=True)
    
if __name__ == "__main__":
    main()
