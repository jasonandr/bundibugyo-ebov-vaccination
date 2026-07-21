import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
from paths import result_path

def plot_stochastic_calibration():
    # Set global font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    with open(result_path("fitted_parameters.json"), "r") as f:
        p = json.load(f)
        
    # Load incidence data
    df = pd.read_csv("BDBV2026-Data/data/insp_sitrep/processed/insp_sitrep__new_confirmed_cases__daily.csv")
    df['new_confirmed_cases'] = pd.to_numeric(df['new_confirmed_cases'], errors='coerce').fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    daily = df.groupby('date')['new_confirmed_cases'].sum().reset_index()
    daily['Day'] = (daily['date'] - daily['date'].min()).dt.days
    
    t_data = daily['Day'].values
    t_data_dates = daily['date'].values
    inc_data = daily['new_confirmed_cases'].values
    
    # Parameters
    N_pop = p.get('N', 10000)
    gamma = p.get('gamma', 1.0 / p.get('infectious_period', 6.0))
    sigma = p.get('sigma', 1.0 / p.get('incubation_period', 8.5))
    incubation_period = 1.0 / sigma
    infectious_period = 1.0 / gamma
    initial_infected = int(p.get('I0', 10) + p.get('E0', 20))
    
    rt_array = p.get('Rt_array', [p['R0']] * int(max(t_data)+10))
    t_rt = np.arange(len(rt_array))
    
    print("Generating network and calibrating tau...")
    G = generate_network(N_pop)
    baseline_tau = calibrate_tau(G, max(rt_array), gamma, num_trials=20)
    
    print("Running stochastic baseline simulations (n=50)...")
    num_runs = 50
    max_sim_time = int(max(t_data)) + 10
    all_incidences = []
    
    for i in range(num_runs):
        daily_inc = simulate_ring_vaccination(
            G, rt_array, baseline_tau, incubation_period, infectious_period,
            uptake=0.0, efficacy=0.0, reporting_rate=0.0,
            initial_infected=initial_infected, max_cases=10000, 
            return_time_series=True, max_sim_time=max_sim_time
        )
        all_incidences.append(daily_inc)
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{num_runs} runs...")
            
    all_incidences = np.array(all_incidences)
    mean_inc = np.mean(all_incidences, axis=0)
    p2_5 = np.percentile(all_incidences, 2.5, axis=0)
    p97_5 = np.percentile(all_incidences, 97.5, axis=0)
    
    # 7-day smoothing of mean for cleaner plotting
    def smooth(arr):
        return pd.Series(arr).rolling(7, min_periods=1, center=True).mean().values
        
    mean_inc_smooth = smooth(mean_inc)
    p2_5_smooth = smooth(p2_5)
    p97_5_smooth = smooth(p97_5)
    
    t_sim = np.arange(max_sim_time + 1)
    t_sim_dates = pd.date_range(start=daily['date'].min(), periods=len(t_sim))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Panel 1: Incidence
    ax1.bar(t_data_dates, inc_data, color='#95A5A6', alpha=0.6, label="Empirical Daily Incidence")
    
    ax1.fill_between(t_sim_dates, p2_5_smooth, p97_5_smooth, color='#E74C3C', alpha=0.3, label="95% Confidence Interval")
    
    ax1.set_ylabel("Daily Confirmed Cases", fontsize=12)
    ax1.set_title("A. Stochastic Network Calibration", loc='left', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Rt
    t_rt_dates = pd.date_range(start=daily['date'].min(), periods=len(t_rt))
    ax2.plot(t_rt_dates, rt_array, color='#2980B9', lw=2.5)
    ax2.axhline(1.0, color='#E74C3C', linestyle=':', lw=2)
    ax2.set_ylabel(r"Effective $R_t$", fontsize=12)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_title(r"B. Dynamic Reproduction Number ($R_t$)", loc='left', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(rt_array)*1.2)
    ax2.set_xlim(t_data_dates.min(), t_data_dates.max())
    
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"stochastic_calibration_{timestamp}.png"
    out_dir = "figures"
    img_path = os.path.join(out_dir, img_name)
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved High-Res Stochastic Calibration to {img_path}")

if __name__ == "__main__":
    plot_stochastic_calibration()
