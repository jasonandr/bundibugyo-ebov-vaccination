import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
import json
import ebola_stochastic_ring as sim
from paths import result_path
from scipy.ndimage import gaussian_filter
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
enh_reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
enh_tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76

def find_valid_seed(start_seed):
    seed = start_seed
    while True:
        base_res = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
            vaccine_effect=0.0, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            max_vaccines=0, base_CFR=BASE_CFR, max_sim_time=90, seed=seed, engine='cpp'
        )
        if base_res[0] * N >= 50:
            return seed, base_res[1] * N
        seed += 1

valid_seeds = []
base_deaths_for_seed = {}

def run_intervention(args):
    cov, eff, seed = args
    base_deaths = base_deaths_for_seed[seed]
    
    comm = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        vaccine_effect=eff, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, community_vax_coverage=cov, community_vax_trigger=1, community_vax_delay=0.0,
        max_sim_time=90, seed=seed, engine='cpp'
    )
    int_deaths = comm[1] * N
    if base_deaths > 0:
        averted = (base_deaths - int_deaths) / base_deaths * 100.0
    else:
        averted = 0.0
    return averted

def plot_enhanced_contour():
    coverages = np.linspace(0.1, 0.8, 15)
    efficacies = np.linspace(0.0, 1.0, 15)
    
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    # We will use 500 valid seeds across the board for all cells
    print("Finding 500 valid baseline seeds...")
    t0 = time.time()
    global valid_seeds, base_deaths_for_seed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(find_valid_seed, range(200000, 200000 + 500*100, 100)))
        
    for seed, deaths in results[:500]:
        valid_seeds.append(seed)
        base_deaths_for_seed[seed] = deaths
    print(f"Found 500 seeds in {time.time()-t0:.1f}s")
    
    args = []
    rep_counts = {}
    for cov in coverages:
        for eff in efficacies:
            # We now just use the 500 valid seeds instead of variable reps, 
            # because 500 valid outbreaks provides much tighter confidence intervals than 2000 raw reps.
            n_reps = 500
            rep_counts[(cov, eff)] = n_reps
            for seed in valid_seeds:
                args.append((cov, eff, seed))
                
    print(f"Running ENHANCED contour map with {len(args)} simulations on {workers} cores...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_intervention, args, chunksize=100))
        
    idx = 0
    Z_averted = np.zeros((len(efficacies), len(coverages)))
    Z_averted_raw = np.zeros((len(efficacies), len(coverages)))
    for i, cov in enumerate(coverages):
        for j, eff in enumerate(efficacies):
            n_reps = rep_counts[(cov, eff)]
            averted = results[idx:idx+n_reps]
            median_val = np.median(averted) if averted else 0
            Z_averted_raw[j, i] = median_val
            Z_averted[j, i] = max(0, median_val)
            idx += n_reps
            
    import pandas as pd
    df_raw = pd.DataFrame(Z_averted_raw, index=np.round(efficacies*100, 1), columns=np.round(coverages*100, 1))
    df_raw.index.name = "Vaccine_Efficacy_Pct"
    os.makedirs("data_and_results", exist_ok=True)
    df_raw.to_csv("data_and_results/fig5_raw_averted_mortality.csv")
            
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    X, Y = np.meshgrid(coverages * 100, efficacies * 100)
    
    Z_averted_smooth = gaussian_filter(Z_averted, sigma=1.2)
    Z_averted_smooth = np.maximum(Z_averted_smooth, 0) # ensure smoothing didn't dip below 0
    
    max_val = max(10, Z_averted_smooth.max())
    
    # Levels from 0 to max_val
    contour = ax.contourf(X, Y, Z_averted_smooth, levels=np.arange(0, max_val + 5, 5), cmap='magma')
    plt.colorbar(contour, ax=ax, label="Median Averted Mortality (%)", ticks=np.arange(0, max_val + 10, 10))
    
    line_contour = ax.contour(X, Y, Z_averted_smooth, levels=np.arange(0, max_val + 10, 10), colors='black', linewidths=1.5)
    ax.clabel(line_contour, inline=True, fontsize=10, fmt='%1.0f%%')
    
    ax.set_title("Figure 5. Sensitivity of Community Mass Vaccination")
    ax.set_xlabel("Community Vaccination Coverage (%)")
    plt.ylabel('Overall Vaccine Effect (%)')
    
    plt.tight_layout()
    path = f"figures/new_analyses/fig5_contour_enhanced.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print(f"CONTOUR={path}")

if __name__ == "__main__":
    plot_enhanced_contour()
