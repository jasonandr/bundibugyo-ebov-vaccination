import sys
import os
import time
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, '/Users/jasonandrews/repos/ebola vaccination modeling/scripts')

from paths import figure_path, result_path
from current_outbreak_data import cumulative_confirmed_cases, cumulative_confirmed_deaths
from ebola_stochastic_ring import generate_network
import ebola_stochastic_ring_cpp as cpp

def main():
    print("Loading empirical outbreak data and fitted parameters...")
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
    cases_inc_ma = pd.Series(cases_inc).rolling(window=7, min_periods=1).mean().values
    
    # Load EpiNow2 Rt data
    epinow_path = 'results/epinow_rt.csv'
    if not os.path.exists(epinow_path):
        epinow_path = '../results/epinow_rt.csv'
    rt_df = pd.read_csv(epinow_path)
    rt_df['date'] = pd.to_datetime(rt_df['date'])
    
    with open(result_path('fitted_parameters.json'), 'r') as f:
        params = json.load(f)
    rt_array = params.get('Rt_array', list(rt_df['median'].values))
    
    N = 100000
    print(f"Generating network for C++ simulation (N={N}, community_variance=160.0)...")
    G = generate_network(N, household_mean=5.2, community_mean=30.0, community_variance=160.0)
    
    # C++ engine setup
    offsets = np.zeros(N + 1, dtype=np.int32)
    edges = []
    for i in range(N):
        offsets[i] = len(edges)
        edges.extend([int(x) for x in G.neighbors(i)])
    offsets[N] = len(edges)
    edges = np.array(edges, dtype=np.int32)
    cpp_engine = cpp.EbolaEngine(N, offsets, edges)
    
    max_sim_time = len(df_daily)
    rt_array_padded = list(rt_array) + [rt_array[-1]] * 30
    
    print("Running 100 C++ cohort engine replicates...")
    n_reps = 100
    sim_incidences = []
    sim_true_rt = []
    
    for rep in range(n_reps):
        res = cpp_engine.run_simulation(
            rt_array_padded, 0.25, 8.5, 6.0, 1, 0.30, 10.0, 0.8,
            [], 0.0, 4.0, 2.0, -1, 100, -1, 0.454, 0.454,
            50, 50, max_sim_time, 0.0, True, 1.0, False, [], -1.0, -1.0, -1.0,
            0.75, 2.0, False, True, 0.0, 0, -1.0, 0.0, 42 + rep, 1.0, False, False, 1.0, 1.0, True
        )
        inc = np.array(res['daily_incidence'][:max_sim_time])
        sim_incidences.append(inc)
        
        num = np.array(res['true_rt_numerator'][:max_sim_time])
        den = np.array(res['true_rt_denominator'][:max_sim_time])
        
        window = 7
        num_smooth = pd.Series(num).rolling(window=window, min_periods=1, center=True).sum().values
        den_smooth = pd.Series(den).rolling(window=window, min_periods=1, center=True).sum().values
        
        rt_vals = np.full_like(num_smooth, np.nan, dtype=float)
        mask = den_smooth > 0
        rt_vals[mask] = num_smooth[mask] / den_smooth[mask]
        sim_true_rt.append(rt_vals)
        
    sim_incidences = np.array(sim_incidences)
    sim_true_rt = np.array(sim_true_rt)
    
    timestamp = int(time.time())
    
    # -------------------------------------------------------------------------
    # Plot 1: Simulated vs Empirical Rt
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    plot_len = min(len(rt_df), max_sim_time)
    dates_plot = rt_df['date'][:plot_len]
    
    # Shaded EpiNow2 CrI
    ax1.fill_between(dates_plot, rt_df['lower_90'][:plot_len], rt_df['upper_90'][:plot_len],
                     color='#2980B9', alpha=0.2, label='EpiNow2 90% CrI')
    ax1.plot(dates_plot, rt_df['median'][:plot_len], color='#2980B9', lw=2.5, linestyle='--',
             label='Empirical Target $R_t$ (EpiNow2 Median)')
    
    # Spaghetti C++ runs
    for i in range(min(40, n_reps)):
        ax1.plot(dates_plot, sim_true_rt[i, :plot_len], color='#E74C3C', alpha=0.12, lw=1)
        
    mean_sim_rt = np.nanmean(sim_true_rt, axis=0)[:plot_len]
    ax1.plot(dates_plot, mean_sim_rt, color='#C0392B', lw=3, label='C++ Simulated Realized $R_t$ (Mean)')
    
    ax1.axhline(1.0, color='black', linestyle=':', lw=1.5, alpha=0.6)
    ax1.set_ylabel("Effective Reproduction Number ($R_t$)", fontsize=14)
    ax1.set_xlabel("Date (2026)", fontsize=14)
    ax1.set_title("C++ Cohort Engine: Simulated Realized $R_t$ vs. Empirical EpiNow2 $R_t$", fontsize=15, pad=12)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', frameon=False, fontsize=12)
    
    plt.tight_layout()
    rt_fig_filename = f"plot_cpp_rt_comparison_{timestamp}.png"
    rt_fig_path = figure_path("polished") / rt_fig_filename
    fig1.savefig(rt_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved Rt comparison figure to {rt_fig_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Simulated vs Empirical Case Counts
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    dates_cases = df_daily['Date'][:max_sim_time]
    
    # Empirical cases bar chart
    ax2.bar(dates_cases, cases_inc[:max_sim_time], color="#5DADE2", width=0.8, alpha=0.5,
            edgecolor='none', label="Empirical Confirmed Cases")
    ax2.plot(dates_cases, cases_inc_ma[:max_sim_time], color="#2980B9", lw=2.5, label="Empirical 7-Day MA")
    
    # C++ simulated cases ensemble
    median_inc = np.median(sim_incidences, axis=0)
    p25_inc = np.percentile(sim_incidences, 2.5, axis=0)
    p975_inc = np.percentile(sim_incidences, 97.5, axis=0)
    
    # Scale simulation incidence to empirical reporting fraction for visual match
    scale_factor = np.sum(cases_inc[:max_sim_time]) / np.sum(median_inc[:max_sim_time])
    scaled_median = median_inc * scale_factor
    scaled_p25 = p25_inc * scale_factor
    scaled_p975 = p975_inc * scale_factor
    
    ax2.fill_between(dates_cases, scaled_p25, scaled_p975, color='#E74C3C', alpha=0.25, label='C++ Simulation 95% CrI')
    ax2.plot(dates_cases, scaled_median, color='#C0392B', lw=3, label='C++ Simulation Median')
    
    ax2.set_ylabel("Daily Incident Cases", fontsize=14)
    ax2.set_xlabel("Date (2026)", fontsize=14)
    ax2.set_title("C++ Cohort Engine: Simulated Outbreak Trajectory vs. Empirical 2026 Data", fontsize=15, pad=12)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper left', frameon=False, fontsize=12)
    
    plt.tight_layout()
    cases_fig_filename = f"plot_cpp_cases_comparison_{timestamp}.png"
    cases_fig_path = figure_path("polished") / cases_fig_filename
    fig2.savefig(cases_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Cases comparison figure to {cases_fig_path}")

    # Copy files to brain artifacts directory for embedding
    brain_dir = Path("/Users/jasonandrews/.gemini/antigravity-ide/brain/b92785c3-f511-471c-a5cd-d92f3cf65e7e")
    import shutil
    shutil.copy(rt_fig_path, brain_dir / rt_fig_filename)
    shutil.copy(cases_fig_path, brain_dir / cases_fig_filename)
    
    with open('/tmp/cpp_val_filenames.json', 'w') as f:
        json.dump({
            'rt_filename': rt_fig_filename,
            'cases_filename': cases_fig_filename,
            'rt_path': str(brain_dir / rt_fig_filename),
            'cases_path': str(brain_dir / cases_fig_filename)
        }, f)

if __name__ == "__main__":
    main()
