import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import json
import ebola_stochastic_ring as sim
from paths import result_path
import time

N = 10000
_G = sim.generate_network(N)
try:
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = np.array(json.load(f)["tau_array"]).tolist()
except:
    TAU_ARRAY = np.linspace(0.12, 0.05, 91).tolist()
try:
    with open(result_path("fitted_parameters.json")) as f:
        params = json.load(f)
except FileNotFoundError:
    params = {}
BASE_CFR = float(params.get("base_CFR", params.get("latest_adjusted_cfr", 0.454)))
VAX_CFR = BASE_CFR * 0.5
VACCINE_EFFECT = 0.5

def find_valid_seed(start_seed):
    seed = start_seed
    while True:
        base_res = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=0, baseline_tau=0.25,
            vaccine_effect=0.0, reporting_rate=[0.3]*91, tracing_coverage=[0.3]*91,
            max_vaccines=0, base_CFR=BASE_CFR, max_sim_time=90, seed=seed, engine='cpp'
        )
        if base_res[0] * N >= 50:
            return seed, base_res[1] * N
        seed += 1

def ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)

def run_intervention(args):
    sig_d0, seed, base_deaths = args
    enh_reporting = ramp(0.7)
    enh_tracing = ramp(0.7)
    
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        vaccine_effect=0.5, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=BASE_CFR*0.5,
        max_sim_time=90, seed=seed, engine='cpp', allow_pep=True,
        immune_delay=sig_d0
    )
    int_deaths = ring[1] * N
    return seed, int_deaths

def main():
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    print("Finding 500 valid baseline seeds...")
    t0 = time.time()
    valid_seeds = []
    base_deaths_for_seed = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(find_valid_seed, range(300000, 300000 + 500*100, 100)))
        
    for seed, deaths in results[:500]:
        valid_seeds.append(seed)
        base_deaths_for_seed[seed] = deaths
    print(f"Found 500 seeds in {time.time()-t0:.1f}s")
    
    rows = []
    for sig_d0, level in [(5.0, "vax_immune_5.0"), (14.0, "vax_immune_14.0")]:
        args = [(sig_d0, seed, base_deaths_for_seed[seed]) for seed in valid_seeds]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_intervention, args, chunksize=50))
        for seed, int_deaths in results:
            rows.append({
                "scenario": "fig5_tornado_immune",
                "level": level,
                "seed": seed,
                "deaths_percent": int_deaths
            })
            
    # Also we need analysis_1_reactive_ring vax_enh_ops! Because load_fig5_data will try to merge against it!
    # Let's just generate it here and add it to the same dataframe!
    args = [(10.0, seed, base_deaths_for_seed[seed]) for seed in valid_seeds]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_intervention, args, chunksize=50))
    for seed, int_deaths in results:
        rows.append({
            "scenario": "analysis_1_reactive_ring",
            "level": "vax_enh_ops",
            "seed": seed,
            "deaths_percent": int_deaths
        })
        
    df = pd.DataFrame(rows)
    df.to_csv("data_and_results/fig5_tornado_immune_results.csv", index=False)
    print("Saved fig5_tornado_immune_results.csv")

if __name__ == "__main__":
    main()
