import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import json
import ebola_stochastic_ring as sim
from paths import result_path
from scipy.ndimage import gaussian_filter

def init_worker():
    global _G, TAU_ARRAY, BASE_CFR, VAX_CFR, VACCINE_EFFECT
    _G = sim.generate_network(10000)
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = json.load(f)["tau_array"]
    BASE_CFR = 0.4539079029615015
    VACCINE_EFFECT = 0.45
    VAX_CFR = BASE_CFR * (1.0 - VACCINE_EFFECT)

def run_contour_point(args):
    detection, tracing, seed = args
    global _G, TAU_ARRAY, BASE_CFR, VAX_CFR, VACCINE_EFFECT
    
    def ramp(target, duration=15, max_time=91):
        return np.linspace(0.3, target, duration).tolist() + [target] * (max_time - duration)
        
    enh_reporting = ramp(detection)
    enh_tracing = ramp(tracing)
    
    # Baseline (no vax, but with these operations)
    no_vax = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=0, baseline_tau=0.25,
        vaccine_effect=0.0, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        max_vaccines=0, base_CFR=BASE_CFR, vax_CFR=BASE_CFR,
        max_sim_time=90, seed=seed, engine='cpp', allow_pep=True
    )
    
    # Ring Vaccination (radius 2)
    ring = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        vaccine_effect=VACCINE_EFFECT, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=VAX_CFR,
        max_sim_time=90, seed=seed, engine='cpp', allow_pep=True
    )
    
    # Community Vaccination (40%)
    comm = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=0, baseline_tau=0.25,
        vaccine_effect=VACCINE_EFFECT, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        base_CFR=BASE_CFR, vax_CFR=VAX_CFR, max_vaccines=0,
        community_vax_coverage=0.4, community_vax_trigger=1, community_vax_delay=0.0,
        community_vax_rollout_days=14.0,
        max_sim_time=90, seed=seed, engine='cpp', allow_pep=True
    )
    
    return no_vax[1], ring[1], comm[1]

def plot_contour():
    detections = np.linspace(0.4, 0.9, 15)
    tracings = np.linspace(0.4, 0.9, 15)
    
    args = []
    n_reps = 1000
    for det in detections:
        for trace in tracings:
            for rep in range(n_reps):
                args.append((det, trace, rep))
                
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"Running Figure 4 contour map with {len(args)} simulations on {workers} cores...")
    
    with Pool(processes=workers, initializer=init_worker) as pool:
        results = pool.map(run_contour_point, args)
        
    idx = 0
    Z_ring = np.zeros((len(tracings), len(detections)))
    Z_comm = np.zeros((len(tracings), len(detections)))
    
    for i, det in enumerate(detections):
        for j, trace in enumerate(tracings):
            chunk = results[idx:idx+n_reps]
            
            averted_ring = []
            averted_comm = []
            
            for no_vax, ring, comm in chunk:
                # Absolute averted mortality per 100k
                # Pop is 10k in this run
                # fraction dead * 100,000
                averted_r = (no_vax - ring) * 100000.0
                averted_c = (no_vax - comm) * 100000.0
                averted_ring.append(averted_r)
                averted_comm.append(averted_c)
                
            Z_ring[j, i] = max(0, np.median(averted_ring))
            Z_comm[j, i] = max(0, np.median(averted_comm))
            idx += n_reps
            
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=150)
    X, Y = np.meshgrid(detections * 100, tracings * 100)
    
    Z_ring_smooth = gaussian_filter(Z_ring, sigma=1.0)
    Z_comm_smooth = gaussian_filter(Z_comm, sigma=1.0)
    
    max_val = max(10, Z_ring_smooth.max(), Z_comm_smooth.max())
    
    # Levels
    levels = np.linspace(0, max_val * 1.05, 12)
    
    c1 = axes[0].contourf(X, Y, Z_ring_smooth, levels=levels, cmap='magma')
    l1 = axes[0].contour(X, Y, Z_ring_smooth, levels=levels[1::2], colors='black', linewidths=1.0, alpha=0.5)
    axes[0].clabel(l1, inline=True, fontsize=9, fmt='%1.0f')
    axes[0].set_title("A: Reactive ring vaccination", fontweight='bold', loc='left')
    
    c2 = axes[1].contourf(X, Y, Z_comm_smooth, levels=levels, cmap='magma')
    l2 = axes[1].contour(X, Y, Z_comm_smooth, levels=levels[1::2], colors='black', linewidths=1.0, alpha=0.5)
    axes[1].clabel(l2, inline=True, fontsize=9, fmt='%1.0f')
    axes[1].set_title("B: Community vaccination (40%)", fontweight='bold', loc='left')
    
    for ax in axes:
        ax.set_xlabel("Index case detection (%)")
        ax.set_ylabel("Contact tracing coverage (%)")
        
    cbar = plt.colorbar(c2, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label("Averted deaths (per 100,000)")
    
    os.makedirs("figures/new_analyses", exist_ok=True)
    path = "figures/new_analyses/fig4_contour_averted.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"CONTOUR={path}")

if __name__ == "__main__":
    plot_contour()
