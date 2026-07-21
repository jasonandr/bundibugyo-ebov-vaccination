import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import sys

# Ensure ebola_stochastic_ring is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau

def run_robustness(dataset_year):
    print(f"\n--- Running Robustness Check for {dataset_year} ---")
    with open(f'data_and_results/historical_params_{dataset_year}.json', 'r') as f:
        params = json.load(f)
        
    R_max = params.get('R_max', 2.0)
    incubation_period = params.get('incubation_period', 8.5)
    infectious_period = params.get('infectious_period', 6.0)
    
    print(f"Loaded R_max = {R_max:.2f}")
    
    N = 10000
    G = generate_network(N)
    
    # Calibrate tau
    print("Calibrating transmission rate (tau) to historical R0...")
    baseline_tau = calibrate_tau(G, R_max, 1.0/infectious_period, num_trials=10)
    print(f"Calibrated tau: {baseline_tau:.4f}")
    
    efficacies = np.linspace(0.2, 1.0, 5)
    reporting_rates = np.linspace(0.2, 0.9, 5)
    
    replicates = 30
    fixed_uptake = 0.8
    fixed_delay = 2.0
    radius = 1 # Test Radius 1 containment
    
    res_cases_grid = np.zeros((len(efficacies), len(reporting_rates)))
    
    for i, eff in enumerate(efficacies):
        for j, rr in enumerate(reporting_rates):
            c_tot = 0.0
            for _ in range(replicates):
                f_inf, f_dead, _ = simulate_ring_vaccination(
                    G, None, baseline_tau, incubation_period, infectious_period,
                    uptake=fixed_uptake, efficacy=eff, reporting_rate=rr, detection_delay=4.0,
                    tracing_delay=fixed_delay, immune_delay=10.0, ring_radius=radius,
                    initial_infected=5, max_cases=10000, max_daily_traces=100
                )
                c_tot += f_inf * N
                
            avg_cases = c_tot / replicates
            res_cases_grid[i, j] = avg_cases
            print(f"  Eff={eff:.2f}, Surv={rr:.2f} -> Avg Final Cases: {avg_cases:.1f}")
            
    # Plotting
    levels = np.linspace(0, min(10000, np.max(res_cases_grid)+100), 10)
    X, Y = np.meshgrid(reporting_rates, efficacies)
    
    plt.rcParams['font.family'] = 'sans-serif'
    
    plt.figure(figsize=(6, 5))
    CS = plt.contourf(X, Y, res_cases_grid, levels=levels, cmap="Reds", alpha=0.9)
    plt.colorbar(CS, label="Total Epidemic Size (Cases)")
    iso = plt.contour(X, Y, res_cases_grid, levels=levels[1::2], colors='white', linewidths=1.0, alpha=0.5)
    plt.clabel(iso, inline=True, fontsize=8, fmt='%1.0f')
    
    plt.title(f"Robustness Check: {dataset_year} Transmission Dynamics", fontsize=12, fontweight='bold')
    plt.xlabel("Surveillance Ascertainment Rate", fontsize=10)
    plt.ylabel("Vaccine Cross-Reactive Efficacy", fontsize=10)
    plt.tight_layout()
    
    timestamp = int(time.time())
    out_path = f"figures/robustness_contour_{dataset_year}_{timestamp}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_robustness("2007")
    run_robustness("2012")
