import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import minimize
import json
import os
import time

from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
from paths import result_path, figure_path
from current_outbreak_data import cumulative_confirmed_cases

def estimate_rt_from_incidence(cases_inc, rolling_window=7):
    # Pre-smooth incidence with a rolling average
    cases_inc = pd.Series(cases_inc).rolling(window=rolling_window, min_periods=1, center=False).mean().values
    
    # Generation time distribution
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, len(cases_inc)+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    prior_mean = 1.5
    prior_sd = 1.0
    prior_shape = (prior_mean / prior_sd)**2
    prior_scale = (prior_sd**2) / prior_mean
    
    Rt_empirical = np.zeros_like(cases_inc)
    
    for t in range(len(cases_inc)):
        t_start = t
        
        def nll(rt_val):
            R = rt_val[0]
            if R <= 0:
                return 1e9
            
            log_prior = gamma.logpdf(R, a=prior_shape, scale=prior_scale)
            log_lik = 0.0
            for s in range(t_start, t + 1):
                Lambda_s = 0.0
                for tau in range(1, s + 1):
                    if tau <= len(w):
                        Lambda_s += cases_inc[s - tau] * w[tau - 1]
                
                mu_s = R * Lambda_s
                if mu_s <= 0:
                    if cases_inc[s] > 0:
                        log_lik += -1e9
                    continue
                
                if cases_inc[s] == 0:
                    log_lik += -mu_s
                else:
                    log_lik += cases_inc[s] * np.log(mu_s) - mu_s
                
            return -(log_lik + log_prior)
        
        res = minimize(nll, x0=[prior_mean], bounds=[(0.01, 20.0)])
        Rt_empirical[t] = res.x[0]

    Rt_smooth = pd.Series(Rt_empirical).rolling(window=3, min_periods=1, center=True).mean().values
    return Rt_smooth

def plot_rt_calibration():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    with open(result_path("fitted_parameters.json"), "r") as f:
        p = json.load(f)
        
    # Load and process empirical incidence
    df = cumulative_confirmed_cases()
    df = df.sort_values('Date')
    
    cases_raw = df['Cases'].values.copy()
    for i in range(len(cases_raw)-2, -1, -1):
        if cases_raw[i] > cases_raw[i+1]:
            cases_raw[i] = cases_raw[i+1]
    df['Cases'] = cases_raw
    
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_daily = pd.DataFrame({'Date': date_range})
    df_daily = pd.merge(df_daily, df, on='Date', how='left')
    df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
    
    cases_cum = df_daily['Cases'].values
    cases_inc_empirical = np.diff(cases_cum, prepend=cases_cum[0])
    
    # Estimate empirical Rt
    print("Estimating Empirical Rt...")
    empirical_rt = estimate_rt_from_incidence(cases_inc_empirical)
    
    # Run simulations
    N_pop = p.get('N', 10000)
    gamma_val = p.get('gamma', 1.0 / p.get('infectious_period', 6.0))
    sigma = p.get('sigma', 1.0 / p.get('incubation_period', 8.5))
    incubation_period = 1.0 / sigma
    infectious_period = 1.0 / gamma_val
    initial_infected = int(p.get('I0', 10) + p.get('E0', 20))
    
    rt_array = p.get('Rt_array', [p['R0']] * len(cases_inc_empirical))
    # Pad rt_array to max_sim_time just in case
    max_sim_time = len(cases_inc_empirical)
    if len(rt_array) < max_sim_time + 10:
        rt_array = list(rt_array) + [rt_array[-1]] * (max_sim_time + 10 - len(rt_array))
        
    print("Generating network and calibrating tau...")
    G = generate_network(N_pop)
    baseline_tau = calibrate_tau(G, max(rt_array), gamma_val, num_trials=20)
    
    print("Running stochastic baseline simulations (n=50)...")
    num_runs = 50
    all_incidences = []
    
    for i in range(num_runs):
        daily_inc = simulate_ring_vaccination(
            G, rt_array, baseline_tau, incubation_period, infectious_period,
            uptake=0.0, efficacy=0.0, reporting_rate=0.0,
            initial_infected=initial_infected, max_cases=10000, 
            return_time_series=True, max_sim_time=max_sim_time - 1
        )
        all_incidences.append(daily_inc[:max_sim_time])
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{num_runs} runs...")
            
    all_incidences = np.array(all_incidences)
    mean_sim_inc = np.mean(all_incidences, axis=0)
    p2_5_sim_inc = np.percentile(all_incidences, 2.5, axis=0)
    p97_5_sim_inc = np.percentile(all_incidences, 97.5, axis=0)
    
    print("Estimating Simulated Rt...")
    simulated_rt = estimate_rt_from_incidence(mean_sim_inc)
    
    # 7-day smoothing for incidence plot
    def smooth(arr):
        return pd.Series(arr).rolling(7, min_periods=1, center=True).mean().values
        
    mean_sim_inc_smooth = smooth(mean_sim_inc)
    p2_5_smooth = smooth(p2_5_sim_inc)
    p97_5_smooth = smooth(p97_5_sim_inc)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    
    # Panel 1: Incidence
    ax1.bar(df_daily['Date'], cases_inc_empirical, color='#95A5A6', alpha=0.5, label="Empirical Daily Incidence")
    ax1.plot(df_daily['Date'], smooth(cases_inc_empirical), color='#34495E', lw=2, label="Empirical (7-day MA)")
    ax1.fill_between(df_daily['Date'], p2_5_smooth, p97_5_smooth, color='#E74C3C', alpha=0.2, label="Simulated (95% UI)")
    ax1.plot(df_daily['Date'], mean_sim_inc_smooth, color='#E74C3C', lw=2, label="Simulated (Mean)")
    
    ax1.set_ylabel("Daily Confirmed Cases", fontsize=12)
    ax1.set_title("A. Incident Cases: Empirical vs. Simulated", loc='left', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Rt
    ax2.plot(df_daily['Date'], empirical_rt, color='#34495E', lw=2.5, label='Empirical $R_t$')
    ax2.plot(df_daily['Date'], simulated_rt, color='#E74C3C', lw=2.5, linestyle='-', label='Simulated $R_t$ (Mean)')
    ax2.plot(df_daily['Date'], rt_array[:max_sim_time], color='#2980B9', lw=1.5, linestyle=':', label='Input $R_t$ Forcing')
    
    ax2.axhline(1.0, color='k', linestyle='--', lw=1.5, alpha=0.5)
    ax2.set_ylabel(r"Effective $R_t$", fontsize=12)
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_title(r"B. Dynamic Reproduction Number ($R_t$)", loc='left', fontsize=14, fontweight='bold')
    ax2.legend(frameon=False, loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(np.nanmax(empirical_rt), np.nanmax(simulated_rt))*1.2)
    ax2.set_xlim(df_daily['Date'].min(), df_daily['Date'].max())
    
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"rt_calibration_{timestamp}.png"
    img_path = figure_path(img_name)
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved plot to {img_path}")
    print(f"FILENAME:{img_path}")

if __name__ == "__main__":
    plot_rt_calibration()
