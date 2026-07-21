import numpy as np
import pandas as pd
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paths import result_path

def run_parameter_sweep():
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
    detection_delay = 4.0 # Days after ONSET before public health detects and rings
    
    print("Generating network...")
    G = generate_network(N)
    
    print("Calibrating baseline tau empirically...")
    R_max = max(rt_array) if rt_array is not None else 1.66
    baseline_tau = calibrate_tau(G, R_max, 1.0/infectious_period, num_trials=30)
    print(f"Empirically calibrated tau for highly clustered network: {baseline_tau:.4f}")
    
    # Run a baseline with 0 coverage and 0 efficacy to compute cases averted
    print("Running baseline unmitigated...")
    baseline_frac, baseline_frac_dead = simulate_ring_vaccination(
        G, rt_array, baseline_tau, incubation_period, infectious_period,
        uptake=0.0, efficacy=0.0, reporting_rate=0.0, detection_delay=detection_delay,
        tracing_delay=0.0, immune_delay=10.0,
        initial_infected=initial_infected, max_cases=10000
    )
    baseline_cases = baseline_frac * N
    baseline_deaths = baseline_frac_dead * N
    print(f"Baseline total cases: {baseline_cases:.1f}, total deaths: {baseline_deaths:.1f}")
    
    # ----------------------------------------------------
    # Sweep 1: Surveillance (Reporting Rate) vs Uptake
    # ----------------------------------------------------
    print("\n--- Running Sweep 1: Surveillance vs Uptake ---")
    reporting_rates = np.linspace(0.10, 0.60, 6)
    uptakes = np.linspace(0.20, 1.0, 9)
    results_sweep1_cases = np.zeros((len(reporting_rates), len(uptakes)))
    results_sweep1_deaths = np.zeros((len(reporting_rates), len(uptakes)))
    
    fixed_efficacy = 0.90
    fixed_tracing_delay = 2.0
    
    for i, rr in enumerate(reporting_rates):
        for j, u in enumerate(uptakes):
            replicates = 5
            frac_infected = 0.0
            frac_dead = 0.0
            for _ in range(replicates):
                f_inf, f_dead = simulate_ring_vaccination(
                    G, rt_array, baseline_tau, incubation_period, infectious_period,
                    uptake=u, efficacy=fixed_efficacy, reporting_rate=rr, detection_delay=detection_delay,
                    tracing_delay=fixed_tracing_delay, immune_delay=10.0,
                    initial_infected=initial_infected, max_cases=10000
                )
                frac_infected += f_inf / replicates
                frac_dead += f_dead / replicates
            
            cases = frac_infected * N
            deaths = frac_dead * N
            averted_cases = baseline_cases - cases
            averted_deaths = baseline_deaths - deaths
            results_sweep1_cases[i, j] = max(0.0, averted_cases)
            results_sweep1_deaths[i, j] = max(0.0, averted_deaths)
            print(f"Reporting: {rr:.2f}, Uptake: {u:.2f} -> {cases:.1f} cases ({averted_cases:.1f} averted), {deaths:.1f} deaths ({averted_deaths:.1f} averted)")
            
    df1_c = pd.DataFrame(results_sweep1_cases, index=np.round(reporting_rates, 2), columns=np.round(uptakes, 2))
    df1_d = pd.DataFrame(results_sweep1_deaths, index=np.round(reporting_rates, 2), columns=np.round(uptakes, 2))
    
    timestamp = int(time.time())
    out_dir = "figures"
    
    # Save Cases Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df1_c, annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title("Cases Averted: Surveillance vs Uptake\n(Fixed Efficacy 90%, Delay 2d, Baseline Cases {:.0f})".format(baseline_cases))
    plt.xlabel("Vaccine Uptake")
    plt.ylabel("Reporting Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_cases_surveillance_vs_uptake_{timestamp}.png"), dpi=150)
    plt.close()
    
    # Save Deaths Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df1_d, annot=True, cmap="Reds", fmt=".0f")
    plt.title("Deaths Averted: Surveillance vs Uptake\n(Fixed Efficacy 90%, Delay 2d, Baseline Deaths {:.0f})".format(baseline_deaths))
    plt.xlabel("Vaccine Uptake")
    plt.ylabel("Reporting Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_deaths_surveillance_vs_uptake_{timestamp}.png"), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # Sweep 2: Tracing Delay vs Uptake at Multiple Detection Fractions
    # ----------------------------------------------------
    print("\n--- Running Sweep 2: Logistics (Tracing Delay) vs Uptake ---")
    delays = np.linspace(0.0, 6.0, 7)
    detection_fractions = [0.132, 0.40, 0.70]
    
    for df_idx, fixed_reporting_rate in enumerate(detection_fractions):
        print(f"\nEvaluating Detection Fraction: {fixed_reporting_rate*100:.1f}%")
        results_sweep2_cases = np.zeros((len(delays), len(uptakes)))
        results_sweep2_deaths = np.zeros((len(delays), len(uptakes)))
        
        for i, d in enumerate(delays):
            for j, u in enumerate(uptakes):
                replicates = 5
                frac_infected = 0.0
                frac_dead = 0.0
                for _ in range(replicates):
                    f_inf, f_dead = simulate_ring_vaccination(
                        G, rt_array, baseline_tau, incubation_period, infectious_period,
                        uptake=u, efficacy=fixed_efficacy, reporting_rate=fixed_reporting_rate, detection_delay=detection_delay,
                        tracing_delay=d, immune_delay=10.0,
                        initial_infected=initial_infected, max_cases=10000
                    )
                    frac_infected += f_inf / replicates
                    frac_dead += f_dead / replicates
                
                cases = frac_infected * N
                deaths = frac_dead * N
                averted_cases = baseline_cases - cases
                averted_deaths = baseline_deaths - deaths
                results_sweep2_cases[i, j] = max(0.0, averted_cases)
                results_sweep2_deaths[i, j] = max(0.0, averted_deaths)
                print(f"Delay: {d:.1f}d, Uptake: {u:.2f} -> {deaths:.1f} deaths ({averted_deaths:.1f} averted)")
                
        df2_c = pd.DataFrame(results_sweep2_cases, index=np.round(delays, 1), columns=np.round(uptakes, 2))
        df2_d = pd.DataFrame(results_sweep2_deaths, index=np.round(delays, 1), columns=np.round(uptakes, 2))
        
        # Save Cases Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df2_c, annot=True, cmap="YlGnBu", fmt=".0f", vmax=3500) 
        plt.title("Cases Averted: Logistics vs Uptake\n(Detection {:.1f}%, Baseline Cases {:.0f})".format(fixed_reporting_rate*100, baseline_cases))
        plt.xlabel("Vaccine Uptake")
        plt.ylabel("Tracing Delay (Days)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_cases_delay_vs_uptake_detect_{int(fixed_reporting_rate*100)}_{timestamp}.png"), dpi=150)
        plt.close()
        
        # Save Deaths Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df2_d, annot=True, cmap="Reds", fmt=".0f", vmax=2000) 
        plt.title("Deaths Averted: Logistics vs Uptake\n(Detection {:.1f}%, Baseline Deaths {:.0f})".format(fixed_reporting_rate*100, baseline_deaths))
        plt.xlabel("Vaccine Uptake")
        plt.ylabel("Tracing Delay (Days)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_deaths_delay_vs_uptake_detect_{int(fixed_reporting_rate*100)}_{timestamp}.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    run_parameter_sweep()
