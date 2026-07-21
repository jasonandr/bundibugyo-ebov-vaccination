import numpy as np
from ebola_stochastic_ring import generate_network
import test_pooled_rt_network as test_rt
import pandas as pd
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence

def run_test():
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
    SHIFT_DAYS = 12
    shifted_rt = np.zeros_like(empirical_rt)
    if len(empirical_rt) > SHIFT_DAYS:
        shifted_rt[:-SHIFT_DAYS] = empirical_rt[SHIFT_DAYS:]
        shifted_rt[-SHIFT_DAYS:] = empirical_rt[-1]
    rt_array = list(shifted_rt) + [shifted_rt[-1]] * 10

    G = test_rt.generate_network(10000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    
    N = len(G.nodes)
    state = np.zeros(N, dtype=int)
    events = []
    
    initial_infected=5
    initial_exposed=20
    inf_mean=6.0
    inc_mean=8.5
    seed_nodes = np.random.choice(N, initial_infected + initial_exposed, replace=False)
    onset_cohort_day0 = []
    
    for i, node in enumerate(seed_nodes):
        if i < initial_infected:
            state[node] = 2 # I
            onset_cohort_day0.append(node)
            rec_day = int(np.round(np.random.gamma(inf_mean, 1.0)))
            events.append((rec_day, 2, node))
        else:
            state[node] = 1 # E
            onset_day = int(np.round(np.random.gamma(inc_mean, 1.0)))
            events.append((onset_day, 1, node))
            
    events.sort(key=lambda x: x[0])
    
    for t in range(1, 40):
        onset_cohort = []
        while events and events[0][0] <= t:
            ev_t, ev_type, node = events.pop(0)
            if ev_type == 1:
                if state[node] == 1:
                    state[node] = 2
                    onset_cohort.append(node)
                    rec_day = t + int(np.round(np.random.gamma(inf_mean, 1.0)))
                    events.append((rec_day, 2, node))
            elif ev_type == 2:
                if state[node] == 2:
                    state[node] = 3
                    
        if onset_cohort:
            target_rt = rt_array[t] if t < len(rt_array) else rt_array[-1]
            expected = len(onset_cohort) * target_rt
            target = int(np.floor(expected))
            if np.random.rand() < (expected - target): target += 1
                
            pool = []
            for node in onset_cohort:
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:
                        pool.append((node, neighbor))
            
            unique_targets = set(t for s, t in pool)
            
            np.random.shuffle(pool)
            actual_infections = 0
            for source, target_node in pool:
                if state[target_node] == 0:
                    state[target_node] = 1
                    exposure_day = t + int(np.round(np.random.uniform(0, inf_mean)))
                    onset_day = exposure_day + int(np.round(np.random.gamma(inc_mean, 1.0)))
                    events.append((onset_day, 1, target_node))
                    actual_infections += 1
                    if actual_infections >= target:
                        break
            
            print(f"Day {t}: Cohort Size {len(onset_cohort)} | Target Infs {target} | Unique Edges Available {len(unique_targets)} | Actual Infs {actual_infections}")
        events.sort(key=lambda x: x[0])

run_test()
