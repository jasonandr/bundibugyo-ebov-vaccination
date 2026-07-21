import numpy as np
import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence

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

empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=1.0)
print(f"Max empirical Rt: {np.max(empirical_rt):.2f}")

# Simulate low incidence from Rt=6
cases_inc_sim = np.zeros_like(cases_inc_empirical)
cases_inc_sim[0] = 5
for i in range(1, len(cases_inc_sim)):
    if i == 14:
        cases_inc_sim[i] = 30
    elif i > 14:
        cases_inc_sim[i] = cases_inc_sim[i-1] * 0.95

simulated_rt = estimate_rt_from_incidence(cases_inc_sim, prior_sd=1.0)
print(f"Max simulated Rt from 5->30 cases: {np.max(simulated_rt):.2f}")
