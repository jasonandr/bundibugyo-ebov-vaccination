import numpy as np
import pandas as pd
from ebola_stochastic_ring import generate_network
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence

def check_fit():
    # same as main logic
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
    
    # Just run a single deterministic sim from the original script to get the incidence
    import test_pooled_rt_network as test_rt
    SHIFT_DAYS = 12
    shifted_rt = np.zeros_like(empirical_rt)
    if len(empirical_rt) > SHIFT_DAYS:
        shifted_rt[:-SHIFT_DAYS] = empirical_rt[SHIFT_DAYS:]
        shifted_rt[-SHIFT_DAYS:] = empirical_rt[-1]
    rt_array = list(shifted_rt) + [shifted_rt[-1]] * 10
    
    G = test_rt.generate_network(10000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    inc = test_rt.run_pooled_sim(G, rt_array, initial_infected=5, initial_exposed=20, max_sim_time=len(cases_inc_empirical)-1)
    
    sim_rt = estimate_rt_from_incidence(inc, prior_sd=0.6)
    
    for i, date in enumerate(df_daily['Date'][:len(cases_inc_empirical)]):
        print(f"{date.strftime('%Y-%m-%d')} | Emp Rt: {empirical_rt[i]:.2f} | Sim Rt: {sim_rt[i]:.2f} | Emp Inc: {cases_inc_empirical[i]:.1f} | Sim Inc: {inc[i]:.1f}")
        
check_fit()
