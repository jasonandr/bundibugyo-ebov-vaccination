import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases

df = cumulative_confirmed_cases()
df = df.sort_values('Date')
cases_cum = df['Cases'].values
cases_inc = np.diff(cases_cum, prepend=cases_cum[0])

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(df['Date'], cases_inc, color='blue', alpha=0.5)
ax.set_title("Empirical Incident Cases")
plt.savefig('../figures/empirical_inc_test.png')
print(f"Empirical cases span {df['Date'].min()} to {df['Date'].max()}")
print(f"Total empirical cases: {cases_cum[-1]}")
