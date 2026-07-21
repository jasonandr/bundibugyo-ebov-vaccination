import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from current_outbreak_data import cumulative_confirmed_cases
import multiprocessing
from functools import partial

# Get Rt array
df = cumulative_confirmed_cases()
df = df.sort_values('Date')
cases_raw = df['Cases'].values.copy()
for i in range(len(cases_raw)-2, -1, -1):
    if cases_raw[i] > cases_raw[i+1]: cases_raw[i] = cases_raw[i+1]
df['Cases'] = cases_raw

date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df_daily = pd.DataFrame({'Date': date_range})
df_daily = pd.merge(df_daily, df, on='Date', how='left')
cases_cum = df_daily['Cases'].interpolate(method='linear').values
cases_inc_empirical = np.diff(cases_cum, prepend=cases_cum[0])

# Using the mocked array from before
empirical_rt = np.linspace(1.5, 6.0, 20).tolist() + np.linspace(6.0, 1.0, 20).tolist() + [1.0]*100
empirical_rt = np.array(empirical_rt)

rt_array = np.zeros(120)
end_len = min(len(empirical_rt), 90)
rt_array[:end_len] = empirical_rt[:end_len]
if end_len < 90:
    rt_array[end_len:90] = empirical_rt[-1]
    
G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=160.0)

base_kwargs = {
    'rt_array': rt_array,
    'max_sim_time': 120,
    'initial_infected': 5,
    'initial_exposed': 20,
    'return_time_series': True,
    'engine': 'cpp'
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

def run_rep(G, kwargs, seed):
    k = kwargs.copy()
    k['seed'] = seed
    res = simulate_ring_vaccination(G, **k)
    return res

if __name__ == '__main__':
    n_reps = 100
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # 1. Spaghetti plot of Rt for "No Intervention"
    results_no_int = pool.map(partial(run_rep, G, scenarios["No Intervention"]), range(n_reps))
    
    plt.figure(figsize=(10, 6))
    for res in results_no_int:
        num = np.array(res['true_rt_numerator'])
        den = np.array(res['true_rt_denominator'])
        # compute 7 day rolling sum to smooth noisy daily estimates
        num_smooth = pd.Series(num).rolling(7, min_periods=1).sum().values
        den_smooth = pd.Series(den).rolling(7, min_periods=1).sum().values
        true_rt = np.zeros_like(num)
        mask = den_smooth > 0
        true_rt[mask] = num_smooth[mask] / den_smooth[mask]
        plt.plot(true_rt, color='blue', alpha=0.1)
        
    plt.plot(rt_array, color='red', linewidth=3, label='Target Rt')
    plt.xlim(0, 90)
    plt.ylim(0, 8)
    plt.title('True Simulated Rt vs Target Rt (No Intervention)')
    plt.legend()
    plt.savefig('../figures/rt_spaghetti_validation.png')
    
    # 2. Deaths table for all 4 scenarios
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
            
            # calculate reductions for each rep relative to baseline mean
            red_list = [(baseline_deaths - d)/baseline_deaths * 100 for d in d_list]
            red_25 = np.percentile(red_list, 2.5)
            red_975 = np.percentile(red_list, 97.5)
            red = f"{red_pct:.1f}% ({red_25:.1f}% - {red_975:.1f}%)"
            
        print(f"| {name} | {mean:.0f} ({p25:.0f} - {p975:.0f}) | {red} |")
