import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os

from ebola_stochastic_ring import simulate_ring_vaccination, generate_network
from paths import result_path

print("Loading parameters...", flush=True)
with open(result_path('rt_calibrated_tau_array.json')) as f:
    tau_array = json.load(f)['tau_array']

N_POP = 100000
I0 = 5
E0 = 0

print("Generating network...", flush=True)
G = generate_network(N_POP, household_mean=5.0, community_mean=10.0, community_variance=25.0)

def run_no_vax(seed):
    return simulate_ring_vaccination(
        G, rt_array=tau_array, ring_radius=0, 
        efficacy=0.0, uptake=0.0, reporting_rate=0.3, detection_delay=4.0, 
        max_sim_time=90, initial_infected=I0, initial_exposed=E0, 
        engine='cpp', seed=seed, base_CFR=0.454, vax_CFR=0.454)
        
def run_vax(seed):
    return simulate_ring_vaccination(
        G, rt_array=tau_array, ring_radius=2, 
        efficacy=0.45, uptake=0.3, reporting_rate=0.3, detection_delay=4.0, tracing_delay=2.0, 
        max_sim_time=90, initial_infected=I0, initial_exposed=E0, 
        engine='cpp', seed=seed, base_CFR=0.454, vax_CFR=0.454)

def run_test():
    reps = 500
    print(f"Running {reps} paired simulations...", flush=True)
    
    with Pool(os.cpu_count() or 4) as pool:
        seeds = list(range(1, reps + 1))
        res_novax = pool.map(run_no_vax, seeds)
        res_vax = pool.map(run_vax, seeds)
        
    novax_deaths_pct = np.array([r[1] for r in res_novax])
    vax_deaths_pct = np.array([r[1] for r in res_vax])
    
    # Calculate absolute deaths (N=100,000)
    novax_deaths = novax_deaths_pct * N_POP
    vax_deaths = vax_deaths_pct * N_POP
    
    # Averted
    averted = []
    for b, v in zip(novax_deaths, vax_deaths):
        if b > 0:
            averted.append((b - v) / b * 100.0)
    averted = np.array(averted)
    
    print("\n=== Test Results (500 Reps) ===")
    print(f"Median No Vax Deaths: {np.median(novax_deaths):.1f} (Mean: {np.mean(novax_deaths):.1f})")
    print(f"Median Vax Deaths:    {np.median(vax_deaths):.1f} (Mean: {np.mean(vax_deaths):.1f})")
    if len(averted) > 0:
        print(f"Median Deaths Averted (%): {np.median(averted):.2f}% (Mean: {np.mean(averted):.2f}%)")
    else:
        print("No paired simulations with deaths > 0 to calculate averted.")

if __name__ == '__main__':
    run_test()
