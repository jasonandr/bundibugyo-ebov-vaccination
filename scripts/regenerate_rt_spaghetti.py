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
from pathlib import Path
from paths import figure_path
import time

def load_data():
    cases_df = cumulative_confirmed_cases()
    cases_df = cases_df.sort_values('Date')
    
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
    emp_df = load_data()
    
    with open('data_and_results/fitted_parameters.json', 'r') as f:
        params = json.load(f)
        
    rt_array = np.array(params.get('Rt_array', []))
    max_sim_time = emp_df['Day'].max() + 10
    if len(rt_array) < max_sim_time:
        rt_array = list(rt_array) + [rt_array[-1]] * int(max_sim_time - len(rt_array))

    print("Generating network (Mean=30, Variance=30, No overdispersion)...")
    G = generate_network(100000, household_mean=5.0, community_mean=30.0, community_variance=30.0)

    print("Running 100 replicates for True Rt extraction...")
    simulated_rt_spaghetti = []
    
    rt_df = pd.read_csv("results/epinow_rt.csv")
    plot_time = min(len(rt_df), max_sim_time)
    for rep in range(100):
        k = {
            'rt_array': rt_array, 'max_sim_time': max_sim_time,
            'initial_infected': 10, 'initial_exposed': 15,
            'baseline_tau': 0.1,
            'detection_delay': 4.0, 'reporting_rate': 0.0,
            'tracing_coverage': 0.0,
            'uptake': 0.0, 
            'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42 + rep
        }
        res = simulate_ring_vaccination(G, **k)
        
        num = np.array(res['true_rt_numerator'][:plot_time])
        den = np.array(res['true_rt_denominator'][:plot_time])
        
        window = 7
        num_smooth = pd.Series(num).rolling(window=window, min_periods=1, center=True).sum().values
        den_smooth = pd.Series(den).rolling(window=window, min_periods=1, center=True).sum().values
        
        rt_vals = np.full_like(num_smooth, np.nan, dtype=float)
        mask = den_smooth > 0
        rt_vals[mask] = num_smooth[mask] / den_smooth[mask]
        
        simulated_rt_spaghetti.append(rt_vals)

    simulated_rt_spaghetti = np.array(simulated_rt_spaghetti)
    mean_rt = np.nanmean(simulated_rt_spaghetti, axis=0)
    
    # Load EpiNow2 Rt
    rt_df = pd.read_csv('results/epinow_rt.csv')
    rt_df['date'] = pd.to_datetime(rt_df['date'])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_time = min(len(rt_df), max_sim_time)
    for i in range(min(50, len(simulated_rt_spaghetti))):
        ax.plot(rt_df['date'][:plot_time], simulated_rt_spaghetti[i, :plot_time], color='#E74C3C', alpha=0.1, lw=1)
        
    ax.plot(rt_df['date'][:plot_time], mean_rt[:plot_time], color='#C0392B', lw=3, label='Simulated True $R_t$ (Mean)')
    
    ax.plot(rt_df['date'][:plot_time], rt_df['median'][:plot_time], color='#2980B9', lw=2.5, linestyle='--', label='Empirical Forcing (EpiNow2 Median $R_t$)')
    ax.fill_between(rt_df['date'][:plot_time], rt_df['lower_90'][:plot_time], rt_df['upper_90'][:plot_time], color='#2980B9', alpha=0.2, label='90% CrI')
    
    ax.axhline(1.0, color='k', linestyle=':', lw=1.5, alpha=0.5)
    ax.set_ylabel(r"Effective $R_t$", fontsize=14)
    ax.set_xlabel("Date (2026)", fontsize=14)
    ax.set_title("Corrected Appendix: Simulated vs. Empirical Effective Reproduction Number", fontsize=16, pad=15)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', frameon=False, fontsize=12)
    
    plt.tight_layout()
    timestamp = int(time.time())
    
    out_file = f'calibration_appendix_fixed_{timestamp}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
