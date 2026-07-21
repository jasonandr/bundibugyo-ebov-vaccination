import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import json
import ebola_stochastic_ring as sim
import time

N = 10000
_G = sim.generate_network(N)
with open('data_and_results/fitted_parameters.json') as f:
    params = json.load(f)
RT_ARRAY = params["Rt_array"]
BASE_CFR = params["base_CFR"]

def ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)

def run_intervention(args):
    sig_d0, seed, base_deaths = args
    enh_reporting = ramp(0.7)
    enh_tracing = ramp(0.7)
    
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=RT_ARRAY, ring_radius=1, baseline_tau=0.25,
        vaccine_effect=0.45, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=BASE_CFR*0.55,
        max_sim_time=90, seed=seed, engine='python', allow_pep=True,
        sigmoidal_k=0.5, sigmoidal_d0=sig_d0
    )
    int_deaths = ring[1] * 100.0
    return seed, int_deaths

def run_base(seed):
    enh_reporting = ramp(0.7)
    enh_tracing = ramp(0.7)
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=RT_ARRAY, ring_radius=1, baseline_tau=0.25,
        vaccine_effect=0.0, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=BASE_CFR,
        max_sim_time=90, seed=seed, engine='python'
    )
    return seed, ring[1] * 100.0

def main():
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    valid_seeds = list(range(20260630, 20260630 + 5000))
    
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        base_results = list(executor.map(run_base, valid_seeds, chunksize=50))
    for seed, deaths in base_results:
        rows.append({
            "scenario": "analysis_1_reactive_ring",
            "level": "no_vax_enh_ops",
            "seed": seed,
            "deaths_percent": deaths,
            "vaccines": 0, "population_size": 10000
        })
        
    for sig_d0, level in [(5.0, "vax_immune_5.0"), (14.0, "vax_immune_14.0")]:
        args = [(sig_d0, seed, 0) for seed in valid_seeds]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_intervention, args, chunksize=50))
        for seed, int_deaths in results:
            rows.append({
                "scenario": "fig5_tornado_immune",
                "level": level,
                "seed": seed,
                "deaths_percent": int_deaths,
                "vaccines": 0, "population_size": 10000
            })
            
    args = [(10.0, seed, 0) for seed in valid_seeds]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_intervention, args, chunksize=50))
    for seed, int_deaths in results:
        rows.append({
            "scenario": "analysis_1_reactive_ring",
            "level": "vax_enh_ops",
            "seed": seed,
            "deaths_percent": int_deaths,
            "vaccines": 0, "population_size": 10000
        })
        
    pd.DataFrame(rows).to_csv("data_and_results/fig5_tornado_immune_results.csv", index=False)
    print("Saved fig5_tornado_immune_results.csv")

if __name__ == "__main__":
    main()
