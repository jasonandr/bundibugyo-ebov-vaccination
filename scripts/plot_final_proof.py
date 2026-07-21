import sys
import os
sys.path.append('scripts')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from ebola_stochastic_ring_old import simulate_ring_vaccination
from ebola_stochastic_ring import generate_network
from current_outbreak_data import cumulative_confirmed_cases
from datetime import datetime, timedelta

def load_data():
    cases_df = cumulative_confirmed_cases()
    cases_df = cases_df.sort_values('Date')
    
    # Enforce monotonicity
    cases_raw = cases_df['Cases'].values.copy()
    for i in range(len(cases_raw)-2, -1, -1):
        if cases_raw[i] > cases_raw[i+1]:
            cases_raw[i] = cases_raw[i+1]
    cases_df['Cases'] = cases_raw
    
    date_range = pd.date_range(start=cases_df['Date'].min(), end=cases_df['Date'].max(), freq='D')
    df_daily = pd.DataFrame({'Date': date_range})
    df_daily = pd.merge(df_daily, cases_df, on='Date', how='left')
    df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
    cases_cum = df_daily['Cases'].values
    cases_inc = np.diff(cases_cum, prepend=cases_cum[0])
    
    df_daily['Incidence_Raw'] = cases_inc
    df_daily['Day'] = (df_daily['Date'] - df_daily['Date'].min()).dt.days
    df_daily['Incidence'] = df_daily['Incidence_Raw'].rolling(window=7, min_periods=1, center=True).mean()
    
    return df_daily

def main():
    print("Loading empirical data and parameters...")
    emp_df = load_data()
    
    with open('data_and_results/fitted_parameters.json', 'r') as f:
        params = json.load(f)
        
    rt_array = np.array(params.get('Rt_array', []))
    max_sim_time = emp_df['Day'].max() + 10
    if len(rt_array) < max_sim_time:
        rt_array = list(rt_array) + [rt_array[-1]] * int(max_sim_time - len(rt_array))

    print("Generating network (Mean=30, Variance=30, No overdispersion)...")
    G = generate_network(100000, household_mean=5.0, community_mean=30.0, community_variance=30.0)

    print("Running 100 replicates (No tracing)...")
    inc_reps = []
    
    for rep in range(100):
        if rep % 10 == 0:
            print(f"Rep {rep}/100")
        k = {
            'rt_array': rt_array, 'max_sim_time': max_sim_time,
            'initial_infected': 10, 'initial_exposed': 15,
            'baseline_tau': 0.1,
            'detection_delay': 4.0, 'reporting_rate': 0.0, # No tracing
            'tracing_coverage': 0.0,
            'uptake': 0.0, 
            'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42 + rep
        }
        res = simulate_ring_vaccination(G, **k)
        inc_reps.append(res['daily_incidence'])

    inc_reps = np.array(inc_reps)
    
    plt.figure(figsize=(10, 6))
    for rep in inc_reps:
        plt.plot(np.arange(len(rep)), rep, color='gray', alpha=0.1)
        
    plt.plot(np.arange(len(inc_reps[0])), np.mean(inc_reps, axis=0), color='black', linewidth=2, label='Simulated Mean (Cohort Engine)')
    
    plt.plot(emp_df['Day'], emp_df['Incidence_Raw'], color='red', alpha=0.3, label='Empirical Raw')
    plt.plot(emp_df['Day'], emp_df['Incidence'], color='red', linewidth=2, linestyle='--', label='Empirical (7d avg)')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Days since Outbreak Start')
    plt.ylabel('Daily Incidence')
    plt.title('Calibrated Baseline Outbreak (No Tracing, No Overdispersion, Cohort Engine)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_file = 'calibration_proof_fixed_2.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
