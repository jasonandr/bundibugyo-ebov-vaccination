import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gamma
from scipy.optimize import minimize
import multiprocessing
from functools import partial
import json

from ebola_stochastic_ring import generate_network
from ebola_stochastic_ring_old import calibrate_tau, simulate_ring_vaccination
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
        t_start = max(0, t - 30) # Small optimization
        
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
        
        res = minimize(nll, x0=[prior_mean], bounds=[(0.01, 6.0)])
        Rt_empirical[t] = res.x[0]

    Rt_smooth = pd.Series(Rt_empirical).rolling(window=3, min_periods=1, center=True).mean().values
    return Rt_smooth

def run_rep(G, kwargs, seed):
    k = kwargs.copy()
    k['seed'] = seed
    return simulate_ring_vaccination(G, **k)

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    
    df = cumulative_confirmed_cases()
    df = df.sort_values('Date')
    
    cases_raw = df['Cases'].values.copy()
    for i in range(len(cases_raw)-2, -1, -1):
        if cases_raw[i] > cases_raw[i+1]:
            cases_raw[i] = cases_raw[i+1]
    df['Cases'] = cases_raw
    with open("../data_and_results/fitted_parameters.json", "r") as f:
        params = json.load(f)
        
    rt_array = params.get("Rt_array", [])
    Rt_smooth = np.array(rt_array)
    
    # EpiNow2 natively back-projects incidence to the time of infection,
    # so we no longer need the 12-day SHIFT_DAYS hack. The Rt array is already aligned.
    target_rt_array = list(Rt_smooth) + [Rt_smooth[-1]] * 10
    
    # Store empirical target Rt for plotting
    tracked_target_rt = target_rt_array
    
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_daily = pd.DataFrame({'Date': date_range})
    df_daily = pd.merge(df_daily, df, on='Date', how='left')
    df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
    
    cases_cum = df_daily['Cases'].values
    cases_inc_empirical = np.diff(cases_cum, prepend=cases_cum[0])
    
    max_sim_time = len(cases_inc_empirical)
    rt_array_padded = list(Rt_smooth) + [Rt_smooth[-1]] * 30
    
    # We will use Rt_smooth as our empirical_rt for plotting
    empirical_rt = Rt_smooth
    
    G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=160.0)
    
    R_max = max(rt_array_padded) if len(rt_array_padded) > 0 else 1.66
    baseline_tau = calibrate_tau(G, R_max, 1.0/6.0, num_trials=10)
    print(f"Calibrated baseline_tau to {baseline_tau:.4f} for R_max {R_max:.4f}")
    
    base_kwargs = {
        'rt_array': rt_array_padded,
        'max_sim_time': max_sim_time + 30,
        'initial_infected': 5,
        'initial_exposed': 5,
        'baseline_tau': baseline_tau,
        'return_time_series': True,
        'engine': 'cohort'
    }
    
    scenarios = {
        "No Intervention": {
            **base_kwargs,
            'detection_delay': 4.0,
            'reporting_rate': 0.8,
            'efficacy': 0.0,
        },
        "Enhanced Ops": {
            **base_kwargs,
            'detection_delay': 2.0,
            'reporting_rate': 1.0,
            'tracing_delay': 1.0,
            'uptake': 0.95,
            'efficacy': 0.0,
        },
        "Enhanced Ops + Ring Vax": {
            **base_kwargs,
            'detection_delay': 2.0,
            'reporting_rate': 1.0,
            'tracing_delay': 1.0,
            'uptake': 0.95,
            'efficacy': 1.0,
            'ring_radius': 1
        },
        "Community Vax (60%)": {
            **base_kwargs,
            'detection_delay': 2.0,
            'reporting_rate': 1.0,
            'tracing_delay': 1.0,
            'uptake': 0.95,
            'efficacy': 1.0,
            'ring_radius': 1,
            'community_vax_trigger': 1,
            'community_vax_delay': 20,
            'community_vax_coverage': 0.6,
            'community_vax_rollout_days': 10
        }
    }
    
    n_reps = 100
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    print("Running simulations...")
    results_no_int = pool.map(partial(run_rep, G, scenarios["No Intervention"]), range(n_reps))
    
    # Process Incidence
    print('RES TYPE:', type(results_no_int[0]))
    print('RES TYPE:', type(results_no_int[0]))
    if type(results_no_int[0]) != dict:
        print('NOT DICT:', results_no_int[0])
    all_incidences = np.array([res['daily_incidence'][:max_sim_time] if type(res) == dict else res[:max_sim_time] for res in results_no_int]).astype(float)
    mean_sim_inc = np.mean(all_incidences, axis=0)
    np.save('../results/simulated_incidences.npy', all_incidences)
    print("Processing tracked True Rt...")
    simulated_rt_spaghetti = []
    for res in results_no_int:
        if type(res) == dict:
            num = np.array(res['true_rt_numerator'][:max_sim_time])
            den = np.array(res['true_rt_denominator'][:max_sim_time])
        else:
            num = np.zeros(max_sim_time)
            den = np.zeros(max_sim_time)
        
        # Smooth the numerator and denominator over a 7-day window to calculate a stable Rt
        window = 7
        num_smooth = pd.Series(num).rolling(window=window, min_periods=1, center=True).sum().values
        den_smooth = pd.Series(den).rolling(window=window, min_periods=1, center=True).sum().values
        
        # Avoid division by zero
        rt_vals = np.full_like(num_smooth, np.nan, dtype=float)
        mask = den_smooth > 0
        rt_vals[mask] = num_smooth[mask] / den_smooth[mask]
        
        simulated_rt_spaghetti.append(rt_vals)
        
    simulated_rt_spaghetti = np.array(simulated_rt_spaghetti)
    simulated_rt_mean = np.nanmean(simulated_rt_spaghetti, axis=0)
    np.save('../results/rt_spaghetti_arrays.npy', simulated_rt_spaghetti)
        
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    
    ax1.bar(df_daily['Date'][:max_sim_time], cases_inc_empirical[:max_sim_time], color='#95A5A6', alpha=0.5, label="Empirical Daily Incidence")
    ax1.plot(df_daily['Date'][:max_sim_time], pd.Series(cases_inc_empirical[:max_sim_time]).rolling(7, min_periods=1, center=True).mean().values, color='#34495E', lw=2, label="Empirical (7-day MA)")
    
    for i in range(n_reps):
        if i == 0:
            ax1.plot(df_daily['Date'][:max_sim_time], all_incidences[i], color='#E74C3C', alpha=0.08, label=f'Simulated Trajectories (n={n_reps})')
        else:
            ax1.plot(df_daily['Date'][:max_sim_time], all_incidences[i], color='#E74C3C', alpha=0.08)
            
    ax1.plot(df_daily['Date'][:max_sim_time], pd.Series(mean_sim_inc).rolling(7, min_periods=1, center=True).mean().values, color='#C0392B', lw=2.5, label="Simulated (Mean)")
    
    ax1.set_ylabel("Daily Confirmed Cases", fontsize=12)
    ax1.set_title("A. Incident Cases: Empirical vs. Simulated", loc='left', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i in range(n_reps):
        if i == 0:
            ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_spaghetti[i], color='#E74C3C', alpha=0.08, label=f'Tracked True $R_t$ (n={n_reps})')
        else:
            ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_spaghetti[i], color='#E74C3C', alpha=0.08)
            
    ax2.plot(df_daily['Date'][:max_sim_time], empirical_rt[:max_sim_time], color='#34495E', lw=2.5, label='Empirical $R_t$')
    ax2.plot(df_daily['Date'][:max_sim_time], simulated_rt_mean[:max_sim_time], color='#C0392B', lw=2.5, linestyle='-', label='Tracked True $R_t$ (Mean)')
    
    ax2.plot(df_daily['Date'][:max_sim_time], Rt_smooth[:max_sim_time], color='#2980B9', lw=2, linestyle=':', label='Input Forcing (EpiNow2 Median $R_t$)')
    
    ax2.axhline(1.0, color='k', linestyle='--', lw=1.5, alpha=0.5)
    ax2.set_ylabel(r"Effective $R_t$", fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=60))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_title(r"B. Dynamic Reproduction Number ($R_t$)", loc='left', fontsize=14, fontweight='bold')
    ax2.legend(frameon=False, loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ymax = max(np.nanmax(empirical_rt), np.nanmax(simulated_rt_mean)) * 1.5
    ymax = min(ymax, 15.0)
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(df_daily['Date'].min(), df_daily['Date'].min() + pd.Timedelta(days=int(max_sim_time-1)))
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    import time
    timestamp = int(time.time())
    img_name = f'final_rt_spaghetti_{timestamp}.png'
    plt.savefig(f'../figures/{img_name}', dpi=300, facecolor='white')
    
    # Save the filename to a text file so we can retrieve it
    with open('latest_plot_filename.txt', 'w') as f:
        f.write(img_name)
    
    print("\nGenerating table...")
    results_dict = {}
    for name, kwargs in scenarios.items():
        if name == "No Intervention":
            res_list = results_no_int
        else:
            res_list = pool.map(partial(run_rep, G, kwargs), range(n_reps))
        
        deaths = [np.sum(r['daily_deaths']) for r in res_list]
        deaths_mean = np.mean(deaths)
        deaths_25 = np.percentile(deaths, 2.5)
        deaths_975 = np.percentile(deaths, 97.5)
        results_dict[name] = (deaths_mean, deaths_25, deaths_975, deaths)
    
    baseline_deaths = results_dict["No Intervention"][0]
    
    print("\n| Scenario | Total Deaths (Mean, 95% CI) | Reduction in Deaths vs Baseline |")
    print("|----------|-----------------------------|---------------------------------|")
    for name, (mean, p25, p975, d_list) in results_dict.items():
        if name == "No Intervention":
            red = "-"
        else:
            red_pct = (baseline_deaths - mean) / baseline_deaths * 100
            red_list = [(baseline_deaths - d)/baseline_deaths * 100 for d in d_list]
            red_25 = np.percentile(red_list, 2.5)
            red_975 = np.percentile(red_list, 97.5)
            red = f"{red_pct:.1f}% ({red_25:.1f}% - {red_975:.1f}%)"
            
        print(f"| {name} | {mean:.0f} ({p25:.0f} - {p975:.0f}) | {red} |")
