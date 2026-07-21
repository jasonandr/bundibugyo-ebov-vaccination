import numpy as np
import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence
import json
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
from paths import result_path

df = cumulative_confirmed_cases()
df = df.sort_values('Date')
cases_raw = df['Cases'].values.copy()
for i in range(len(cases_raw)-2, -1, -1):
    if cases_raw[i] > cases_raw[i+1]: cases_raw[i] = cases_raw[i+1]
df['Cases'] = cases_raw
date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df_daily = pd.DataFrame({'Date': date_range})
df_daily = pd.merge(df_daily, df, on='Date', how='left')
dates = df_daily['Date'].values
cases_cum = df_daily['Cases'].interpolate(method='linear').values
cases_inc_empirical = np.diff(cases_cum, prepend=cases_cum[0])

empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=0.6)

SHIFT_DAYS = 12
shifted_rt = np.zeros_like(empirical_rt)
shifted_rt[:-SHIFT_DAYS] = empirical_rt[SHIFT_DAYS:]
shifted_rt[-SHIFT_DAYS:] = empirical_rt[-1]

G = generate_network(10000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
with open(result_path("fitted_parameters.json"), "r") as f:
    p = json.load(f)
rt_array = list(shifted_rt) + [shifted_rt[-1]] * 10
baseline_tau = calibrate_tau(G, max(rt_array), p.get('gamma', 1.0/6.0), num_trials=5)

all_inc = []
for _ in range(20):
    daily_inc = simulate_ring_vaccination(
        G, rt_array, baseline_tau, 8.5, 6.0,
        uptake=0.0, efficacy=0.0, reporting_rate=0.0,
        initial_infected=5, initial_exposed=20, max_cases=10000, 
        return_time_series=True, max_sim_time=len(cases_inc_empirical)-1, engine='cpp'
    )
    all_inc.append(daily_inc)
mean_sim_inc = np.mean(all_inc, axis=0)
simulated_rt_mean = estimate_rt_from_incidence(mean_sim_inc, prior_sd=0.6)

for i, d in enumerate(dates[:len(simulated_rt_mean)]):
    if pd.to_datetime(d).month == 6 and pd.to_datetime(d).day >= 15:
        print(f"{str(d)[:10]}: Forcing Rt = {shifted_rt[i]:.2f}, Simulated Rt = {simulated_rt_mean[i]:.2f}, Empirical Rt = {empirical_rt[i]:.2f}, Sim Inc = {mean_sim_inc[i]:.2f}")
    if pd.to_datetime(d).month == 7 and pd.to_datetime(d).day <= 5:
        print(f"{str(d)[:10]}: Forcing Rt = {shifted_rt[i]:.2f}, Simulated Rt = {simulated_rt_mean[i]:.2f}, Empirical Rt = {empirical_rt[i]:.2f}, Sim Inc = {mean_sim_inc[i]:.2f}")

