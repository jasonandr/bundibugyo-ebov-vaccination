import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from concurrent.futures import ProcessPoolExecutor
import ebola_stochastic_ring as sim
import matplotlib
from paths import result_path
import time

matplotlib.use('Agg')

def setup_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

N = 10000
_G = sim.generate_network(N)
try:
    import json
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = np.array(json.load(f)["tau_array"]).tolist()
except:
    TAU_ARRAY = np.linspace(0.12, 0.05, 91).tolist()
try:
    import json
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

def run_risk_compensation_point(args):
    mult, eff, seed = args
    base_deaths = base_deaths_for_seed[seed]
    
    comm = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        vaccine_effect=eff, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        community_vax_coverage=0.50, community_vax_trigger=1, community_vax_delay=0.0,
        community_vax_rollout_days=14.0,
        risk_compensation_multiplier=mult, base_CFR=BASE_CFR,
        max_sim_time=90, seed=seed, engine='cpp'
    )
    int_deaths = comm[1] * N
    
    if base_deaths > 0:
        averted = (base_deaths - int_deaths) / base_deaths * 100.0
    else:
        averted = 0.0
    return averted

def plot_risk_compensation():
    setup_style()
    csv_path = "data_and_results/fig8_raw_averted_mortality.csv"
    multipliers = np.linspace(1.0, 2.0, 15)
    efficacies = np.linspace(0.0, 1.0, 15)
    
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path} instead of running simulations...")
        df_raw = pd.read_csv(csv_path, index_col=0)
        Z_change = df_raw.values
    else:
        n_reps = 500
        
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
        for mult in multipliers:
            for eff in efficacies:
                for seed in valid_seeds:
                    args.append((mult, eff, seed))
                    
        print(f"Running risk compensation contour map with {len(args)} simulations on {workers} cores...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_risk_compensation_point, args, chunksize=100))
            
        idx = 0
        Z_change = np.zeros((len(efficacies), len(multipliers)))
        for i, mult in enumerate(multipliers):
            for j, eff in enumerate(efficacies):
                change = results[idx:idx+n_reps]
                Z_change[j, i] = np.median(change) if change else 0
                idx += n_reps
                
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_raw = pd.DataFrame(Z_change, index=np.round(efficacies*100, 1), columns=np.round(multipliers, 2))
        df_raw.index.name = "Vaccine_Efficacy_Pct"
        df_raw.to_csv(csv_path)
            
    fig, ax = plt.subplots(figsize=(5.1, 4.25), dpi=300)
    X, Y = np.meshgrid(multipliers, efficacies * 100)
    
    from scipy.ndimage import gaussian_filter
    Z_change_smooth = gaussian_filter(Z_change, sigma=1.2)
    
    data_min = np.nanmin(Z_change_smooth)
    data_max = np.nanmax(Z_change_smooth)
    lower = min(-60, 5 * np.floor(data_min / 5))
    upper = max(60, 5 * np.ceil(data_max / 5))
    levels = np.arange(lower, upper + 5, 5)
    
    contour = ax.contourf(X, Y, Z_change_smooth, levels=levels, cmap='coolwarm_r', extend='both', alpha=0.9)
    cbar = plt.colorbar(contour, ax=ax, label="Median averted mortality (%)")
    cbar.ax.tick_params(labelsize=7)
    
    line_levels = [level for level in [-40, -20, -10, 0, 20, 40, 60] if lower <= level <= upper]
    line_contour = ax.contour(X, Y, Z_change_smooth, levels=line_levels, colors='#111827', linewidths=1.0, alpha=0.85)
    ax.clabel(line_contour, inline=True, fontsize=7, fmt='%1.0f%%')
    zero_contour = ax.contour(X, Y, Z_change_smooth, levels=[0], colors='#111827', linewidths=1.5)
    ax.clabel(zero_contour, inline=True, fontsize=7, fmt={0: "0%"})
    
    ax.set_xlabel("Behavioral risk compensation (multiplier on contact rate)")
    ax.set_ylabel("Overall vaccine effect (%)")
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(0, 100)
    
    ax.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout(pad=0.6)
    path = f"figures/polished/fig8_risk_comp.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    pdf_path = f"figures/polished/fig8_risk_comp.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"RISK_COMP={path}")

if __name__ == "__main__":
    plot_risk_compensation()
