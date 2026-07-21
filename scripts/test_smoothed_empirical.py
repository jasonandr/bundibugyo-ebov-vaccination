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

# Try 14-day smoothing on incidence BEFORE estimating Rt
cases_inc_smooth_14 = pd.Series(cases_inc_empirical).rolling(window=14, min_periods=1, center=True).mean().values
rt_smooth_14 = estimate_rt_from_incidence(cases_inc_smooth_14, prior_sd=1.0)
print("Max Rt with 14-day smoothed incidence:", np.max(rt_smooth_14))

# Try 21-day smoothing
cases_inc_smooth_21 = pd.Series(cases_inc_empirical).rolling(window=21, min_periods=1, center=True).mean().values
rt_smooth_21 = estimate_rt_from_incidence(cases_inc_smooth_21, prior_sd=1.0)
print("Max Rt with 21-day smoothed incidence:", np.max(rt_smooth_21))
