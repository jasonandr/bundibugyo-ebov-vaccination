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

def ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)

def run_ops(args):
    scenario, level, seed, reporting, tracing = args
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=0, baseline_tau=0.25,
        vaccine_effect=0.0, reporting_rate=reporting, tracing_coverage=tracing,
        max_vaccines=0, base_CFR=BASE_CFR, max_sim_time=90, seed=seed, engine='cpp'
    )
    return scenario, level, seed, ring[1] * N

def main():
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    # Read seeds from fig5_tornado_immune_results.csv
    df = pd.read_csv("data_and_results/fig5_tornado_immune_results.csv")
    valid_seeds = df["seed"].unique()
    
    args = []
    for seed in valid_seeds:
        args.append(("analysis_1_reactive_ring", "no_vax_base_ops", seed, [0.3]*91, [0.3]*91))
        args.append(("analysis_1_reactive_ring", "no_vax_enh_ops", seed, ramp(0.7), ramp(0.7)))
        
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_ops, args, chunksize=50))
        
    rows = []
    for scenario, level, seed, deaths in results:
        rows.append({
            "scenario": scenario,
            "level": level,
            "seed": seed,
            "deaths_percent": deaths,
            "vaccines": 0,
            "population_size": 10000
        })
        
    new_df = pd.DataFrame(rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv("data_and_results/fig5_tornado_immune_results.csv", index=False)
    print("Saved updated fig5_tornado_immune_results.csv")

if __name__ == "__main__":
    main()
