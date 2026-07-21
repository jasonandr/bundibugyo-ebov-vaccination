import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
import json
import time
import os
from paths import result_path

def run_efficacy_sweep():
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
    
    efficacies = np.linspace(0.0, 1.0, 7)
    reporting_rates = np.linspace(0.1, 0.9, 7)
    
    replicates = 150
    fixed_uptake = 0.8
    fixed_delay = 2.0
    
    out_dir = "figures"
    timestamp = int(time.time())
    
    for radius in [1, 2]:
        print(f"\nSweeping for Ring Radius {radius}...")
        res_cases_grid = np.zeros((len(efficacies), len(reporting_rates)))
        res_deaths_grid = np.zeros((len(efficacies), len(reporting_rates)))
        
        for i, eff in enumerate(efficacies):
            for j, rr in enumerate(reporting_rates):
                c_tot, d_tot = 0.0, 0.0
                for _ in range(replicates):
                    f_inf, f_dead, _ = simulate_ring_vaccination(
                        G, rt_array, baseline_tau, incubation_period, infectious_period,
                        uptake=fixed_uptake, efficacy=eff, reporting_rate=rr, detection_delay=4.0,
                        tracing_delay=fixed_delay, immune_delay=10.0, ring_radius=radius,
                        initial_infected=10, max_cases=10000, max_daily_traces=100
                    )
                    c_tot += f_inf * N
                    d_tot += f_dead * N
                    
                cases_averted = baseline_cases - (c_tot / replicates)
                deaths_averted = baseline_deaths - (d_tot / replicates)
                
                res_cases_grid[i, j] = max(0.0, cases_averted)
                res_deaths_grid[i, j] = max(0.0, deaths_averted)
                print(f"  Eff={eff:.2f}, Surv={rr:.2f} -> Deaths Averted: {deaths_averted:.1f}")
                
        levels_cases = np.linspace(0, max(np.max(res_cases_grid), 100), 12)
        levels_deaths = np.linspace(0, max(np.max(res_deaths_grid), 100), 12)
        
        X, Y = np.meshgrid(reporting_rates, efficacies)
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        
        # Cases Contour
        plt.figure(figsize=(8, 6))
        CS = plt.contourf(X, Y, res_cases_grid, levels=levels_cases, cmap="viridis", alpha=0.9)
        plt.colorbar(CS, label="Cases Averted")
        iso = plt.contour(X, Y, res_cases_grid, levels=levels_cases[1::2], colors='white', linewidths=1.0, alpha=0.5)
        plt.clabel(iso, inline=True, fontsize=8, fmt='%1.0f')
        
        plt.title(f"Cases Averted (Radius {radius})", fontsize=14, fontweight='bold')
        plt.xlabel("Surveillance Ascertainment Rate", fontsize=12)
        plt.ylabel("Vaccine Cross-Reactive Efficacy", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"contour_efficacy_cases_radius_{radius}_{timestamp}.png"), dpi=300)
        plt.close()
        
        # Deaths Contour
        plt.figure(figsize=(8, 6))
        CS2 = plt.contourf(X, Y, res_deaths_grid, levels=levels_deaths, cmap="magma", alpha=0.9)
        plt.colorbar(CS2, label="Deaths Averted")
        iso2 = plt.contour(X, Y, res_deaths_grid, levels=levels_deaths[1::2], colors='white', linewidths=1.0, alpha=0.5)
        plt.clabel(iso2, inline=True, fontsize=8, fmt='%1.0f')
        
        plt.title(f"Deaths Averted (Radius {radius})", fontsize=14, fontweight='bold')
        plt.xlabel("Surveillance Ascertainment Rate", fontsize=12)
        plt.ylabel("Vaccine Cross-Reactive Efficacy", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"contour_efficacy_deaths_radius_{radius}_{timestamp}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    run_efficacy_sweep()
