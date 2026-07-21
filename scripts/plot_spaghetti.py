import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import datetime
import os
import networkx as nx

import ebola_stochastic_ring as sim

_G = None
def init_worker():
    global _G
    _G = sim.generate_network(100000)

import json
from paths import result_path

def run_paired_simulation(seed):
    global _G
    G = _G
    
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = json.load(f)["tau_array"]
    
    # 1. Base Ops No Vax (Radius 0)
    no_vax = sim.simulate_ring_vaccination(
        G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=0, 
        efficacy=0.0, uptake=0.0, immune_delay=10.0,
        reporting_rate=0.7, detection_delay=4.0, tracing_delay=2.0,
        max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
    )
    
    # 2. Radius 1
    vax_r1 = sim.simulate_ring_vaccination(
        G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=1, 
        efficacy=0.8, uptake=0.7, immune_delay=10.0,
        reporting_rate=0.7, detection_delay=4.0, tracing_delay=2.0,
        max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
    )
    
    # 3. Radius 2 (Base Ops)
    vax_r2 = sim.simulate_ring_vaccination(
        G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, 
        efficacy=0.8, uptake=0.7, immune_delay=10.0,
        reporting_rate=0.7, detection_delay=4.0, tracing_delay=2.0,
        max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
    )
    
    # 4. Optimal Radius 2
    opt_vax_r2 = sim.simulate_ring_vaccination(
        G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, 
        efficacy=0.8, uptake=0.7, immune_delay=10.0,
        reporting_rate=0.9, detection_delay=2.0, tracing_delay=2.0,
        max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
    )
    
    return np.array(no_vax), np.array(vax_r1), np.array(vax_r2), np.array(opt_vax_r2)

if __name__ == "__main__":
    n_sims = int(os.environ.get("SPAGHETTI_REPS", "10000"))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    print(f"Running {n_sims} paired simulations across 4 scenarios with {workers} workers...")
    seeds = list(range(1, n_sims + 1))
    
    with Pool(processes=workers, initializer=init_worker) as pool:
        results = pool.map(run_paired_simulation, seeds)
        
    print("Simulations complete. Generating plot...")
    
    max_days = 90
    days = np.arange(max_days + 1)
    
    # Calculate cumulative deaths averted
    def calc_averted(no_vax, vax):
        return np.cumsum(no_vax[:max_days+1]) - np.cumsum(vax[:max_days+1])
        
    averted_r1 = np.array([calc_averted(r[0], r[1]) for r in results])
    averted_r2 = np.array([calc_averted(r[0], r[2]) for r in results])
    averted_opt = np.array([calc_averted(r[0], r[3]) for r in results])
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    def plot_scenario(averted_data, color, label):
        final_averted = averted_data[:, -1]
        q25 = np.percentile(final_averted, 25)
        q75 = np.percentile(final_averted, 75)
        in_iqr = (final_averted >= q25) & (final_averted <= q75)
        
        # Plot up to 500 lines that fall within the IQR to prevent massive file sizes and visual clutter
        plotted_count = 0
        for i in range(n_sims):
            if in_iqr[i]:
                ax.plot(days, averted_data[i], color=color, alpha=0.02, linewidth=1)
                plotted_count += 1
                if plotted_count >= 500:
                    break
        
        # Plot Median
        median = np.median(averted_data, axis=0)
        ax.plot(days, median, color=color, linewidth=2.5, label=label)

    plot_scenario(averted_r1, '#3498db', "Radius 1 (Base Ops)")
    plot_scenario(averted_r2, '#e74c3c', "Radius 2 (Base Ops)")
    plot_scenario(averted_opt, '#2ecc71', "Radius 2 (Optimal Ops)")
    
    ax.set_xlabel("Days since outbreak start")
    ax.set_ylabel("Cumulative Deaths Averted")
    ax.set_title(f"Spaghetti Plot: Cumulative Deaths Averted over Time\n(IQR subset of N={n_sims:,} pairs)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    
    plt.tight_layout()
    plot_path = f"figures/new/spaghetti_multi_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
