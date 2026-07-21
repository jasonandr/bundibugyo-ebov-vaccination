import pandas as pd
import json
from current_outbreak_data import cumulative_confirmed_cases
df = cumulative_confirmed_cases()
df = df.sort_values('Date')
with open("../data_and_results/fitted_parameters.json", "r") as f:
    params = json.load(f)
rt_array = params.get("Rt_array", [])
date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df_daily = pd.DataFrame({'Date': date_range})
df_daily = pd.merge(df_daily, df, on='Date', how='left')
for i in range(len(rt_array)):
    if i < len(df_daily):
        print(f"{df_daily['Date'].iloc[i].strftime('%Y-%m-%d')}: {rt_array[i]}")
