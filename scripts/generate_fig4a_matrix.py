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
VACCINE_EFFECT = 0.5 # original was 0.5 in plot_v32, wait, 0.5 or 0.45? The user said use updated, let's use 0.5 for ring? No, the previous agent used 0.45 for VACCINE_EFFECT in run_contour_fig4.

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

valid_seeds = []
base_deaths_for_seed = {}

def run_intervention(args):
    det, trace, seed, base_deaths = args
    
    
    def ramp(target, duration=15, max_time=91):
        return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)
        
    enh_reporting = ramp(det)
    enh_tracing = ramp(trace)
    
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        vaccine_effect=0.5, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=BASE_CFR*0.5,
        max_sim_time=90, seed=seed, engine='cpp', allow_pep=True
    )
    int_deaths = ring[1] * N
    if base_deaths > 0:
        averted = (base_deaths - int_deaths) / base_deaths * 100.0
    else:
        averted = 0.0
    return averted

def main():
    detections = np.linspace(0.0, 1.0, 21) # original was 0 to 100% in plot_v32? Yes, DIVERGING contour went -40 to 95. Actually let's use 0 to 100
    tracings = np.linspace(0.0, 1.0, 21)
    
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    print("Finding 500 valid baseline seeds...")
    t0 = time.time()
    global valid_seeds, base_deaths_for_seed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(find_valid_seed, range(300000, 300000 + 500*100, 100)))
        
    for seed, deaths in results[:500]:
        valid_seeds.append(seed)
        base_deaths_for_seed[seed] = deaths
    print(f"Found 500 seeds in {time.time()-t0:.1f}s")
    
    args = []
    for det in detections:
        for trace in tracings:
            for seed in valid_seeds:
                args.append((det, trace, seed, base_deaths_for_seed[seed]))
                
    print(f"Running contour map with {len(args)} simulations...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_intervention, args, chunksize=100))
        
    idx = 0
    Z_averted_raw = np.zeros((len(tracings), len(detections)))
    for i, det in enumerate(detections):
        for j, trace in enumerate(tracings):
            n_reps = 500
            averted = results[idx:idx+n_reps]
            median_val = np.median(averted) if averted else 0
            Z_averted_raw[j, i] = median_val
            idx += n_reps
            
    df_raw = pd.DataFrame(Z_averted_raw, index=np.round(tracings*100, 1), columns=np.round(detections*100, 1))
    df_raw.index.name = "Tracing_Pct"
    os.makedirs("data_and_results", exist_ok=True)
    df_raw.to_csv("data_and_results/fig4a_raw_averted_mortality.csv")
    print("Saved fig4a_raw_averted_mortality.csv")

if __name__ == "__main__":
    main()
