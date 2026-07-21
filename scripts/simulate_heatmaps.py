import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
import json
import time
import os
from paths import result_path

def run_heatmap_sweep():
    with open(result_path('fitted_parameters.json'), 'r') as f:
        params = json.load(f)
        
    rt_array = params.get('Rt_array', None)
    incubation_period = params.get('incubation_period', 8.5)
    infectious_period = params.get('infectious_period', 6.0)
    
    N = 10000
    G = generate_network(N)
    R_max = max(rt_array) if rt_array is not None else 1.66
    baseline_tau = calibrate_tau(G, R_max, 1.0/infectious_period, num_trials=30)
    
    print("Running baseline unmitigated...")
    baseline_frac, baseline_frac_dead, _ = simulate_ring_vaccination(
        G, rt_array, baseline_tau, incubation_period, infectious_period,
        uptake=0.0, efficacy=0.0, reporting_rate=0.0, detection_delay=4.0,
        tracing_delay=0.0, immune_delay=10.0, ring_radius=1,
        initial_infected=10, max_cases=10000
    )
    baseline_cases = baseline_frac * N
    baseline_deaths = baseline_frac_dead * N
    
    uptakes = np.linspace(0.4, 1.0, 7)
    tracing_delays = np.linspace(1.0, 7.0, 7)
    reporting_rate = 0.132
    
    res_cases_grid = np.zeros((len(uptakes), len(tracing_delays)))
    res_deaths_grid = np.zeros((len(uptakes), len(tracing_delays)))
    
    replicates = 5
    
    print(f"Sweeping heatmap for reporting_rate={reporting_rate}...")
    for i, u in enumerate(uptakes):
        for j, td in enumerate(tracing_delays):
            c_tot, d_tot = 0.0, 0.0
            for _ in range(replicates):
                f_inf, f_dead, _ = simulate_ring_vaccination(
                    G, rt_array, baseline_tau, incubation_period, infectious_period,
                    uptake=u, efficacy=0.90, reporting_rate=reporting_rate, detection_delay=4.0,
                    tracing_delay=td, immune_delay=10.0, ring_radius=1,
                    initial_infected=10, max_cases=10000, max_daily_traces=100
                )
                c_tot += f_inf * N
                d_tot += f_dead * N
                
            cases_averted = baseline_cases - (c_tot / replicates)
            deaths_averted = baseline_deaths - (d_tot / replicates)
            
            res_cases_grid[i, j] = max(0.0, cases_averted)
            res_deaths_grid[i, j] = max(0.0, deaths_averted)
            
    timestamp = int(time.time())
    out_dir = "figures"
    
    levels_cases = np.linspace(0, max(np.max(res_cases_grid), 100), 12)
    levels_deaths = np.linspace(0, max(np.max(res_deaths_grid), 100), 12)
    
    X, Y = np.meshgrid(tracing_delays, uptakes)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Cases Contour
    plt.figure(figsize=(8, 6))
    CS = plt.contourf(X, Y, res_cases_grid, levels=levels_cases, cmap="viridis", alpha=0.9)
    plt.colorbar(CS, label="Cases Averted")
    iso = plt.contour(X, Y, res_cases_grid, levels=levels_cases[1::2], colors='white', linewidths=1.0, alpha=0.5)
    plt.clabel(iso, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(f"Cases Averted (Ascertainment {reporting_rate*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Tracing & Logistics Delay (Days)", fontsize=12)
    plt.ylabel("Community Vaccine Uptake", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"contour_cases_delay_vs_uptake_detect_13_{timestamp}.png"), dpi=300)
    plt.close()
    
    # Deaths Contour
    plt.figure(figsize=(8, 6))
    CS2 = plt.contourf(X, Y, res_deaths_grid, levels=levels_deaths, cmap="magma", alpha=0.9)
    plt.colorbar(CS2, label="Deaths Averted")
    iso2 = plt.contour(X, Y, res_deaths_grid, levels=levels_deaths[1::2], colors='white', linewidths=1.0, alpha=0.5)
    plt.clabel(iso2, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(f"Deaths Averted (Ascertainment {reporting_rate*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Tracing & Logistics Delay (Days)", fontsize=12)
    plt.ylabel("Community Vaccine Uptake", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"contour_deaths_delay_vs_uptake_detect_13_{timestamp}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_heatmap_sweep()
