import pandas as pd
import sys
sys.path.append('.')
from current_outbreak_data import cumulative_confirmed_cases

df = cumulative_confirmed_cases()
df = df.sort_values('Date')
cases_raw = df['Cases'].values.copy()
for i in range(len(cases_raw)-2, -1, -1):
    if cases_raw[i] > cases_raw[i+1]:
        cases_raw[i] = cases_raw[i+1]
df['Cases'] = cases_raw

date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df_daily = pd.DataFrame({'Date': date_range})
df_daily = df_daily.merge(df[['Date', 'Cases']], on='Date', how='left')
df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
df_daily['Incidence'] = df_daily['Cases'].diff().fillna(0)

print(df_daily.head(60))
