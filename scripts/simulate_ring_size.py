import numpy as np
import pandas as pd
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paths import result_path

def run_ring_size_sweep():
    try:
        with open(result_path('fitted_parameters.json'), 'r') as f:
            params = json.load(f)
        rt_array = params.get('Rt_array', None)
        incubation_period = params.get('incubation_period', 8.5)
        infectious_period = params.get('infectious_period', 6.0)
    except FileNotFoundError:
        rt_array = None
        incubation_period = 8.5
        infectious_period = 6.0
    
    N = 10000
    initial_infected = 10
    detection_delay = 4.0
    
    print("Generating network...")
    G = generate_network(N)
    
    print("Calibrating baseline tau empirically...")
    R_max = max(rt_array) if rt_array is not None else 1.66
    baseline_tau = calibrate_tau(G, R_max, 1.0/infectious_period, num_trials=30)
    print(f"Empirically calibrated tau for highly clustered network: {baseline_tau:.4f}")
    
    print("Running baseline unmitigated...")
    baseline_frac, baseline_frac_dead, _ = simulate_ring_vaccination(
        G, rt_array, baseline_tau, incubation_period, infectious_period,
        uptake=0.0, efficacy=0.0, reporting_rate=0.0, detection_delay=detection_delay,
        tracing_delay=0.0, immune_delay=10.0, ring_radius=1,
        initial_infected=initial_infected, max_cases=10000
    )
    baseline_cases = baseline_frac * N
    baseline_deaths = baseline_frac_dead * N
    print(f"Baseline total cases: {baseline_cases:.1f}, total deaths: {baseline_deaths:.1f}")
    
    # ----------------------------------------------------
    # Sweep: Ring Size (Radius 1 vs 2) vs Ascertainment
    # ----------------------------------------------------
    print("\n--- Evaluating Ring Size Impact ---")
    detection_fractions = np.linspace(0.1, 0.8, 8)
    
    # Empirical 2018 DRC Anchors
    fixed_uptake = 0.84 
    fixed_tracing_delay = 2.0
    fixed_efficacy = 0.90
    
    results = []
    
    for df in detection_fractions:
        print(f"\nDetection Fraction: {df*100:.1f}%")
        for radius in [1, 2]:
            replicates = 8
            frac_dead_total = 0.0
            vaccines_total = 0.0
            for _ in range(replicates):
                f_inf, f_dead, vax = simulate_ring_vaccination(
                    G, rt_array, baseline_tau, incubation_period, infectious_period,
                    uptake=fixed_uptake, efficacy=fixed_efficacy, reporting_rate=df, detection_delay=detection_delay,
                    tracing_delay=fixed_tracing_delay, immune_delay=10.0, ring_radius=radius,
                    initial_infected=initial_infected, max_cases=10000
                )
                frac_dead_total += f_dead / replicates
                vaccines_total += vax / replicates
                
            deaths = frac_dead_total * N
            averted = baseline_deaths - deaths
            results.append({
                "Detection Fraction": df,
                "Ring Radius": f"Radius {radius} ({'Contacts' if radius==1 else 'Contacts-of-Contacts'})",
                "Deaths Averted": max(0.0, averted),
                "Total Vaccines": vaccines_total
            })
            print(f"  Radius {radius}: {vaccines_total:.0f} vaccines used -> {averted:.1f} deaths averted")
            
    df_res = pd.DataFrame(results)
    
    timestamp = int(time.time())
    out_dir = "figures"
    
    levels_cases = np.linspace(0, max(np.max(res_cases_grid), 100), 12)
    levels_deaths = np.linspace(0, max(np.max(res_deaths_grid), 100), 12)
    
    # Cases Averted Contour
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    X, Y = np.meshgrid(tracing_delays, uptakes)
    CS = plt.contourf(X, Y, res_cases_grid.T, levels=levels_cases, cmap="viridis", alpha=0.9)
    plt.colorbar(CS, label="Cases Averted")
    
    # Add iso-clines
    iso = plt.contour(X, Y, res_cases_grid.T, levels=levels_cases[1::2], colors='white', linewidths=1.0, alpha=0.5)
    plt.clabel(iso, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(f"Cases Averted (Ascertainment {reporting_rate*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Tracing & Logistics Delay (Days)", fontsize=12)
    plt.ylabel("Community Vaccine Uptake", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_cases_delay_vs_uptake_detect_{int(reporting_rate*100)}_{timestamp}.png"), dpi=300)
    plt.close()
    
    # Deaths Averted Contour
    plt.figure(figsize=(8, 6))
    CS2 = plt.contourf(X, Y, res_deaths_grid.T, levels=levels_deaths, cmap="magma", alpha=0.9)
    plt.colorbar(CS2, label="Deaths Averted")
    
    # Add iso-clines
    iso2 = plt.contour(X, Y, res_deaths_grid.T, levels=levels_deaths[1::2], colors='white', linewidths=1.0, alpha=0.5)
    plt.clabel(iso2, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(f"Deaths Averted (Ascertainment {reporting_rate*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Tracing & Logistics Delay (Days)", fontsize=12)
    plt.ylabel("Community Vaccine Uptake", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_deaths_delay_vs_uptake_detect_{int(reporting_rate*100)}_{timestamp}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_ring_size_sweep()
