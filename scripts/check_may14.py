import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
import numpy as np

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

for i, date in enumerate(df_daily['Date']):
    print(f"{date.strftime('%Y-%m-%d')}: {cases_inc_empirical[i]:.1f} cases (cum: {cases_cum[i]:.0f})")
    if i > 15: break
