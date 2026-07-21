import pandas as pd
import numpy as np
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
empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=0.6)

print(f"Min empirical Rt: {np.min(empirical_rt)}")
for i, r in enumerate(empirical_rt):
    if r < 1.0:
        print(f"Empirical Rt crosses 1.0 on: {df_daily['Date'][i]}")
        break
