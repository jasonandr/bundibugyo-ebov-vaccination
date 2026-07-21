import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ebola_stochastic_ring as sim
import matplotlib
from scipy.ndimage import gaussian_filter1d

matplotlib.use('Agg')

COLORS = {
    "No Vaccine": "#4B5563",         # Charcoal
    "Reactive Ring": "#4F6D7A",      # Blue-grey
    "Community Vax": "#1F9D8A",      # Teal
    "Hybrid": "#D96B4A"              # Coral
}

import multiprocessing as mp
from functools import partial

def run_single_dose(i, strat, G, coverage=0.0):
    tau_array = np.linspace(0.12, 0.05, 91).tolist()
    enh_reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
    enh_tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76
    
    if strat == "no_vax":
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.0, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=0.0, max_sim_time=90, engine='cpp'
        )
    elif strat == "ring":
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.40, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=0.0, max_sim_time=90, engine='cpp'
        )
    elif strat == "comm":
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.40, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=coverage, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_vaccines=0, max_sim_time=90, engine='cpp'
        )
    else: # hybrid
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.40, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=coverage, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_sim_time=90, engine='cpp'
        )
        
    return {"Total Deaths": r[1] * 10000, "Total Vaccinated": r[2]}

def run_dose_simulations_parallel(n_reps=200):
    G = sim.generate_network(10000)
    coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    comm_results = {c: [] for c in coverages}
    hybrid_results = {c: [] for c in coverages}

    with mp.Pool(mp.cpu_count() - 1) as pool:
        func = partial(run_single_dose, strat="no_vax", G=G)
        no_vax_res = list(pool.imap_unordered(func, range(n_reps), chunksize=25))
        
        func = partial(run_single_dose, strat="ring", G=G)
        ring_res = list(pool.imap_unordered(func, range(n_reps), chunksize=25))
        
        for c in coverages:
            func = partial(run_single_dose, strat="comm", G=G, coverage=c)
            comm_results[c] = list(pool.imap_unordered(func, range(n_reps), chunksize=25))
            
            func = partial(run_single_dose, strat="hybrid", G=G, coverage=c)
            hybrid_results[c] = list(pool.imap_unordered(func, range(n_reps), chunksize=25))
            
    baseline_deaths = np.array([r["Total Deaths"] for r in no_vax_res])
    baseline_median = np.median(baseline_deaths)
    
    return baseline_median, ring_res, comm_results, hybrid_results

def calculate_metrics(res_list, baseline_median):
    deaths = np.array([r["Total Deaths"] for r in res_list])
    doses = np.array([r["Total Vaccinated"] for r in res_list]) / 10000.0 * 100000.0 # scale to per 100k
    
    mortality_reduction = (baseline_median - deaths) / baseline_median * 100
    return {
        "Doses Median": np.median(doses),
        "Doses p25": np.percentile(doses, 25),
        "Doses p75": np.percentile(doses, 75),
        "Mortality Red Median": np.median(mortality_reduction),
        "Mortality Red p25": np.percentile(mortality_reduction, 25),
        "Mortality Red p75": np.percentile(mortality_reduction, 75),
    }

def plot_dose_impact():
    baseline_median, ring_res, comm_results, hybrid_results = run_dose_simulations_parallel(500)
    
    # Process Ring
    r_stats = calculate_metrics(ring_res, baseline_median)
    
    # Process Comm and Hybrid
    c_df = []
    h_df = []
    coverages = sorted(list(comm_results.keys()))
    for c in coverages:
        cs = calculate_metrics(comm_results[c], baseline_median)
        cs["Coverage"] = c
        c_df.append(cs)
        
        hs = calculate_metrics(hybrid_results[c], baseline_median)
        hs["Coverage"] = c
        h_df.append(hs)
        
    c_df = pd.DataFrame(c_df)
    h_df = pd.DataFrame(h_df)
    
    fig, axA = plt.subplots(figsize=(10, 6), dpi=150)
    
    # --- PANEL A: Impact Frontier ---
    axA.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Plot Community
    axA.plot(c_df["Doses Median"], c_df["Mortality Red Median"], color=COLORS["Community Vax"], linewidth=2, label="Community vaccination")
    axA.fill_between(c_df["Doses Median"], c_df["Mortality Red p25"], c_df["Mortality Red p75"], color=COLORS["Community Vax"], alpha=0.15)
    axA.scatter(c_df["Doses Median"], c_df["Mortality Red Median"], color=COLORS["Community Vax"], s=40, zorder=3)
    
    # Plot Hybrid
    axA.plot(h_df["Doses Median"], h_df["Mortality Red Median"], color=COLORS["Hybrid"], linewidth=2, label="Hybrid")
    axA.fill_between(h_df["Doses Median"], h_df["Mortality Red p25"], h_df["Mortality Red p75"], color=COLORS["Hybrid"], alpha=0.15)
    axA.scatter(h_df["Doses Median"], h_df["Mortality Red Median"], color=COLORS["Hybrid"], s=40, zorder=3)
    
    # Plot Ring (single point)
    axA.errorbar(r_stats["Doses Median"], r_stats["Mortality Red Median"], 
                 yerr=[[r_stats["Mortality Red Median"] - r_stats["Mortality Red p25"]], [r_stats["Mortality Red p75"] - r_stats["Mortality Red Median"]]],
                 xerr=[[r_stats["Doses Median"] - r_stats["Doses p25"]], [r_stats["Doses p75"] - r_stats["Doses Median"]]],
                 color=COLORS["Reactive Ring"], fmt='o', capsize=0, zorder=5)
                 
    # Labels
    for c in [0.2, 0.4, 0.6, 0.8]:
        row = c_df[c_df["Coverage"] == c].iloc[0]
        axA.text(row["Doses Median"], row["Mortality Red Median"] + 3, f"{int(c*100)}%", ha='center', color=COLORS["Community Vax"], fontsize=9)
        
    # Direct Line Labels
    c_last = c_df.iloc[-1]
    axA.text(c_last["Doses Median"] + 2000, c_last["Mortality Red Median"], "Community\nvaccination", va='center', color=COLORS["Community Vax"], fontweight='bold')
    
    h_last = h_df.iloc[-1]
    axA.text(h_last["Doses Median"] + 2000, h_last["Mortality Red Median"] - 4, "Hybrid", va='center', color=COLORS["Hybrid"], fontweight='bold')
    
    axA.text(r_stats["Doses Median"] + 2000, r_stats["Mortality Red Median"] + 2, "Reactive ring", va='center', color=COLORS["Reactive Ring"], fontweight='bold')
    
    axA.set_xlabel("Vaccine courses delivered per 100,000 population")
    axA.set_ylabel("Mortality reduction (%)")
    axA.set_title("Dose-Impact Frontier", loc="left", fontweight='bold', fontsize=14)
    axA.grid(axis='y', linestyle='-', alpha=0.1)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = "figures/new_analyses/fig6_dose_impact_v2.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"FIG6={path}")

if __name__ == "__main__":
    plot_dose_impact()
