import numpy as np
import pandas as pd
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paths import result_path

def run_dynamic_sweep():
    # Load Rt array
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
        
    # Load Empirical Detection array
    try:
        with open(result_path('detection_array.json'), 'r') as f:
            detection_data = json.load(f)
        empirical_detection = detection_data['detection_array']
    except FileNotFoundError:
        print("Error: data_and_results/detection_array.json not found. Run estimate_detection.py first.")
        return
        
    N = 10000
    initial_infected = 10
    detection_delay = 4.0
    
    print("Generating network...")
    G = generate_network(N)
    
    print("Calibrating baseline tau empirically...")
    R_max = max(rt_array) if rt_array is not None else 1.66
    baseline_tau = calibrate_tau(G, R_max, 1.0/infectious_period, num_trials=30)
    print(f"Empirically calibrated tau: {baseline_tau:.4f}")
    
    print("Running baseline unmitigated...")
    baseline_frac, baseline_frac_dead, _ = simulate_ring_vaccination(
        G, rt_array, baseline_tau, incubation_period, infectious_period,
        uptake=0.0, efficacy=0.0, reporting_rate=0.0, detection_delay=detection_delay,
        tracing_delay=0.0, immune_delay=10.0, ring_radius=1,
        initial_infected=initial_infected, max_cases=10000
    )
    baseline_deaths = baseline_frac_dead * N
    print(f"Baseline total deaths: {baseline_deaths:.1f}")
    
    # ----------------------------------------------------
    # Scenarios: Static Poor (13.2%) vs Empirical Dynamic
    # ----------------------------------------------------
    print("\n--- Evaluating Dynamic vs Static Surveillance ---")
    fixed_uptake = 0.84 
    fixed_tracing_delay = 2.0
    fixed_efficacy = 0.90
    
    scenarios = [
        {"name": "Static Poor (13.2%)", "rr": 0.132},
        {"name": "Empirical Dynamic Scale-up", "rr": empirical_detection}
    ]
    
    results = []
    
    for sc in scenarios:
        print(f"\nScenario: {sc['name']}")
        for radius in [1, 2]:
            replicates = 30
            print(f"  Running Radius {radius} ({replicates} replicates)...")
            for _ in range(replicates):
                f_inf, f_dead, vax = simulate_ring_vaccination(
                    G, rt_array, baseline_tau, incubation_period, infectious_period,
                    uptake=fixed_uptake, efficacy=fixed_efficacy, reporting_rate=sc['rr'], detection_delay=detection_delay,
                    tracing_delay=fixed_tracing_delay, immune_delay=10.0, ring_radius=radius,
                    initial_infected=initial_infected, max_cases=10000, max_daily_traces=100
                )
                deaths = f_dead * N
                averted = baseline_deaths - deaths
                results.append({
                    "Surveillance Scenario": sc['name'],
                    "Ring Radius": f"Radius {radius}",
                    "Deaths Averted": max(0.0, averted),
                    "Total Vaccines": vax
                })
            
    df_res = pd.DataFrame(results)
    
    timestamp = int(time.time())
    out_dir = "figures"
    
    # Plot 1: Deaths Averted (Violin chart)
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df_res, x="Surveillance Scenario", y="Deaths Averted", hue="Ring Radius", palette="mako", cut=0, inner="quartile")
    plt.title("Epidemic Control: Variance in Deaths Averted\n(Uptake 84%, Efficacy 90%, Max 100 Traces/Day)")
    plt.ylabel("Deaths Averted")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"violin_dynamic_deaths_{timestamp}.png"), dpi=300)
    plt.close()
    
    # Plot 2: Total Vaccines
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df_res, x="Surveillance Scenario", y="Total Vaccines", hue="Ring Radius", palette="flare", cut=0, inner="quartile")
    plt.title("Logistical Cost: Variance in Vaccine Demand\n(Uptake 84%, Efficacy 90%, Max 100 Traces/Day)")
    plt.ylabel("Total Vaccines Administered")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"violin_dynamic_vaccines_{timestamp}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_dynamic_sweep()
