import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import minimize
import json
import os
import time

from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from paths import result_path, figure_path
from current_outbreak_data import cumulative_confirmed_cases

def estimate_rt_from_incidence(cases_inc, rolling_window=7, prior_sd=0.6):
    cases_inc = pd.Series(cases_inc).rolling(window=rolling_window, min_periods=1, center=False).mean().values
    
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, len(cases_inc)+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    prior_mean = 1.5
    prior_shape = (prior_mean / prior_sd)**2
    prior_scale = (prior_sd**2) / prior_mean
    
    Rt_empirical = np.zeros_like(cases_inc)
    
    for t in range(len(cases_inc)):
        t_start = t
        
        def nll(rt_val):
            R = rt_val[0]
            if R <= 0: return 1e9
            
            log_prior = gamma.logpdf(R, a=prior_shape, scale=prior_scale)
            log_lik = 0.0
            for s in range(t_start, t + 1):
                Lambda_s = 0.0
                for tau in range(1, s + 1):
                    if tau <= len(w):
                        Lambda_s += cases_inc[s - tau] * w[tau - 1]
                
                mu_s = R * Lambda_s
                if mu_s <= 0:
                    if cases_inc[s] > 0: log_lik += -1e9
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

def fast_rt(cases_inc, window=7):
    N_sims, T = cases_inc.shape
    kernel = np.ones(window) / window
    cases_inc_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full')[:T], axis=1, arr=cases_inc)
    
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, T+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    # Using the exact same prior for regularization
    alpha_prior = (1.5 / 0.6)**2
    beta_prior = alpha_prior / 1.5
    
    Rt = np.zeros((N_sims, T))
    for t in range(1, T):
        Lambda = np.zeros(N_sims)
        for tau in range(1, t+1):
            if tau <= len(w):
                Lambda += cases_inc_smooth[:, t-tau] * w[tau-1]
        
        Rt[:, t] = (alpha_prior + cases_inc_smooth[:, t]) / (beta_prior + Lambda)
        
    return Rt

def main():
    plt.rcParams['font.family'] = 'sans-serif'
    
    with open(result_path("fitted_parameters.json"), "r") as f:
        p = json.load(f)
        
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
    
    # We use a slightly tighter prior (0.6) to gracefully handle the massive reporting dumps
    empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=0.6)
    
    # Shift Rt to the left by 12 days to align transmission forcing with actual transmission dates
    SHIFT_DAYS = 12
    shifted_rt = np.zeros_like(empirical_rt)
    if len(empirical_rt) > SHIFT_DAYS:
        shifted_rt[:-SHIFT_DAYS] = empirical_rt[SHIFT_DAYS:]
        shifted_rt[-SHIFT_DAYS:] = empirical_rt[-1] # Pad end
    else:
        shifted_rt = empirical_rt
    
    print("Generating network (10,000 nodes for quick test)...")
    # For a rapid test we use 10,000. 100,000 takes a bit longer.
    G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    
    gamma_val = p.get('gamma', 1.0 / 6.0)
    incubation_period = 8.5
    infectious_period = 6.0
    initial_infected = 5
    initial_exposed = 20
    
    max_sim_time = len(cases_inc_empirical)
    rt_array = list(shifted_rt) + [shifted_rt[-1]] * 10
    
    print("Calibrating tau...")
    baseline_tau = 0.25 # (G, max(rt_array), gamma_val, num_trials=20)
    
    print("Running 100 stochastic simulations...")
    num_runs = 100
    all_incidences = []
    
    for i in range(num_runs):
        daily_inc = simulate_ring_vaccination(
            G, rt_array, baseline_tau, incubation_period, infectious_period
            uptake=0.0, efficacy=0.0, reporting_rate=0.0
            initial_infected=initial_infected, initial_exposed=initial_exposed, max_cases=10000, 
            return_time_series=True, max_sim_time=max_sim_time - 1, engine='cpp'
        )
        all_incidences.append(daily_inc[:max_sim_time])
        if (i+1) % 20 == 0:
            print(f"  Completed {i+1}/{num_runs} runs...")
            
    all_incidences = np.array(all_incidences).astype(float)
    mean_sim_inc = np.mean(all_incidences, axis=0)
    
    simulated_rt_mean = estimate_rt_from_incidence(mean_sim_inc, prior_sd=0.6)
    simulated_rt_spaghetti = fast_rt(all_incidences)
    
    for i in range(simulated_rt_spaghetti.shape[0]):
        simulated_rt_spaghetti[i] = pd.Series(simulated_rt_spaghetti[i]).rolling(window=3, min_periods=1, center=True).mean().values
    
    # PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    
    ax1.bar(df_daily['Date'][:max_sim_time], cases_inc_empirical[:max_sim_time], color='#95A5A6', alpha=0.5, label="Empirical Daily Incidence")
    ax1.plot(df_daily['Date'][:max_sim_time], pd.Series(cases_inc_empirical[:max_sim_time]).rolling(7, min_periods=1, center=True).mean().values, color='#34495E', lw=2, label="Empirical (7-day MA)")
    
    for i in range(num_runs):
        if i == 0:
            ax1.plot(df_daily['Date'][:max_sim_time], all_incidences[i], color='#E74C3C', alpha=0.08, label='Simulated Trajectories (n=100)')
        else:
            ax1.plot(df_daily['Date'][:max_sim_time], all_incidences[i], color='#E74C3C', alpha=0.08)
            
    ax1.plot(df_daily['Date'][:max_sim_time], pd.Series(mean_sim_inc).rolling(7, min_periods=1, center=True).mean().values, color='#C0392B', lw=2.5, label="Simulated (Mean)")
    
    ax1.set_ylabel("Daily Confirmed Cases", fontsize=12)
    ax1.set_title("A. Incident Cases: Empirical vs. Simulated", loc='left', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Rt
    for i in range(num_runs):
        if i == 0:
            ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_spaghetti[i], color='#E74C3C', alpha=0.08, label='Simulated $R_t$ Trajectories (n=100)')
        else:
            ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_spaghetti[i], color='#E74C3C', alpha=0.08)
            
    ax2.plot(df_daily['Date'][:max_sim_time], empirical_rt[:max_sim_time], color='#34495E', lw=2.5, label='Empirical $R_t$')
    ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_mean[:max_sim_time], color='#C0392B', lw=2.5, linestyle='-', label='Simulated $R_t$ (Mean)')
    
    ax2.plot(df_daily['Date'][:max_sim_time], shifted_rt[:max_sim_time], color='#2980B9', lw=2, linestyle=':', label='Input Forcing (Shifted Empirical $R_t$)')
    
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
    
    ymax = max(np.nanmax(empirical_rt), np.nanmax(simulated_rt_mean))*1.5
    ymax = min(ymax, 6.0)
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(df_daily['Date'].min(), df_daily['Date'].min() + pd.Timedelta(days=int(max_sim_time-1)))
    
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"rt_test_shifted_{timestamp}.png"
    img_path = figure_path(img_name)
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved plot to {img_path}")
    print(f"FILENAME:{img_path}")

if __name__ == "__main__":
    main()
