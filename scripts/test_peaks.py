import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
import numpy as np

# Recreate what plot_rt_calibration_test.py does to find the dates
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

print("Start Date:", dates[0])
print("Day 12:", dates[12])
print("Day 42:", dates[42])
