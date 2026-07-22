import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import os
import subprocess

from paths import figure_path, result_path
from current_outbreak_data import cumulative_confirmed_cases

def estimate_rt():
    print("Preparing empirical incidence data for EpiNow2...")
    df = cumulative_confirmed_cases()
    df = df.sort_values('Date')
    
    # Enforce monotonicity before interpolation
    cases_raw = df['Cases'].values.copy()
    for i in range(len(cases_raw)-2, -1, -1):
        if cases_raw[i] > cases_raw[i+1]:
            cases_raw[i] = cases_raw[i+1]
    df['Cases'] = cases_raw
    
    # Create daily grid
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_daily = pd.DataFrame({'Date': date_range})
    df_daily = pd.merge(df_daily, df, on='Date', how='left')
    
    # Interpolate missing cumulative values
    df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
    
    cases_cum = df_daily['Cases'].values
    cases_inc_raw = np.diff(cases_cum, prepend=cases_cum[0]) # Daily incidence
    
    # Ensure cases are positive integers
    cases_inc_raw = np.maximum(0, np.round(cases_inc_raw)).astype(int)
    
    df_daily['Cases'] = cases_inc_raw
    
    # Export for R
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    df_daily[['Date', 'Cases']].to_csv('../data/empirical_incidence.csv', index=False)
    
    # Run EpiNow2
    print("Running EpiNow2 in R (this may take a few minutes)...")
    try:
        subprocess.run(['Rscript', 'estimate_rt_epinow.R'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"EpiNow2 failed: {e}")
        return
        
    print("EpiNow2 complete. Loading results...")
    
    # Read Rt estimates
    rt_df = pd.read_csv('../results/epinow_rt.csv')
    
    # Merge Rt estimates with our daily date grid
    rt_df['date'] = pd.to_datetime(rt_df['date'])
    df_daily = pd.merge(df_daily, rt_df, left_on='Date', right_on='date', how='left')
    
    # Fill missing (EpiNow2 might not give estimates for the very end/beginning)
    df_daily['median'] = df_daily['median'].fillna(method='ffill').fillna(method='bfill')
    df_daily['lower_90'] = df_daily['lower_90'].fillna(method='ffill').fillna(method='bfill')
    df_daily['upper_90'] = df_daily['upper_90'].fillna(method='ffill').fillna(method='bfill')
    
    Rt_smooth = df_daily['median'].values
    Rt_lower = df_daily['lower_90'].values
    Rt_upper = df_daily['upper_90'].values
    
    print(f"Rt Estimation Complete!")
    print(f"Empirical Rt at Day 0: {Rt_smooth[0]:.2f}")
    print(f"Empirical Rt at End: {Rt_smooth[-1]:.2f}")
    print(f"Empirical Rt range: {np.nanmin(Rt_smooth):.2f}-{np.nanmax(Rt_smooth):.2f}")
    
    # Save parameters
    try:
        with open(result_path('fitted_parameters.json'), 'r') as f:
            out_params = json.load(f)
    except FileNotFoundError:
        out_params = {}
        
    out_params.update({
        'Rend': float(Rt_smooth[-1]),
        'Rt_min': float(np.nanmin(Rt_smooth)),
        'Rt_max': float(np.nanmax(Rt_smooth)),
        'Rt_final': float(Rt_smooth[-1]),
        'latest_data_date': str(df_daily['Date'].max().date()),
        'latest_confirmed_cases': int(round(cases_cum[-1])),
        'Rt_array': Rt_smooth.tolist()
    })
    
    with open(result_path('fitted_parameters.json'), 'w') as f:
        json.dump(out_params, f, indent=4)
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # TOP PLOT: Incidence
    ax1.bar(df_daily['Date'], cases_inc_raw, color='#34495E', alpha=0.6, label='Raw Incident Cases (Boxes)')
    cases_inc_smooth = pd.Series(cases_inc_raw).rolling(window=7, min_periods=1, center=False).mean().values
    ax1.plot(df_daily['Date'], cases_inc_smooth, 'b-', linewidth=2, label='7-Day Moving Average')
    ax1.set_ylabel('Cases / Day', fontsize=12)
    ax1.set_title('Ebola Outbreak Incidence & Bayesian Rt (EpiNow2)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # BOTTOM PLOT: Rt
    ax2.plot(df_daily['Date'], Rt_smooth, 'r-', linewidth=2, label='EpiNow2 $R_t$ (Median)')
    ax2.fill_between(df_daily['Date'], Rt_lower, Rt_upper, color='r', alpha=0.2, label='90% Credible Interval')
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Effective Reproduction Number ($R_t$)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax1.set_xlim(df_daily['Date'].min() - pd.Timedelta(days=1), df_daily['Date'].max() + pd.Timedelta(days=1))
    ax2.set_ylim(0, max(6.0, np.max(Rt_upper)*1.1))
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"rt_estimation_{timestamp}.png"
    img_path = figure_path(img_name)
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved plot to {img_path}")
    print(f"FILENAME:{img_name}")

if __name__ == "__main__":
    estimate_rt()
