import numpy as np
import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
import pymc as pm

def estimate_rt_from_incidence(incidence, mean_si=15.3, std_si=9.3, prior_mean=1.5, prior_sd=1.0):
    shape = (mean_si / std_si)**2
    scale = (std_si**2) / mean_si
    t_max = 60
    t = np.arange(1, t_max + 1)
    w = (t**(shape-1) * np.exp(-t/scale))
    w = w / np.sum(w)
    
    T = len(incidence)
    lambda_t = np.zeros(T)
    for t_idx in range(1, T):
        for tau in range(1, min(t_idx, t_max) + 1):
            lambda_t[t_idx] += incidence[t_idx - tau] * w[tau - 1]
            
    with pm.Model() as model:
        Rt = pm.LogNormal('Rt', mu=np.log(prior_mean), sigma=prior_sd, shape=T)
        mu = Rt * lambda_t
        mu = pm.math.switch(mu < 1e-5, 1e-5, mu)
        obs = pm.Poisson('obs', mu=mu, observed=incidence)
        idata = pm.sample(1000, tune=500, return_inferencedata=True, progressbar=False)
        
    rt_mean = idata.posterior['Rt'].mean(dim=['chain', 'draw']).values
    return rt_mean

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
cases_inc = np.diff(cases_cum, prepend=cases_cum[0])
rt = estimate_rt_from_incidence(cases_inc)

print("Cases over first 90 days:", cases_inc[:90])
print("Rt over first 90 days:", rt[:90])
