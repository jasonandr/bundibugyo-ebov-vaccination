import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination, calibrate_tau
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence

def run_sim_track_rt(G, target_rt_array):
    N = len(G.nodes)
    state = np.zeros(N, dtype=int)
    events = []
    
    initial_infected = 5
    initial_exposed = 20
    inc_mean = 8.5
    inf_mean = 6.0
    
    seed_nodes = np.random.choice(N, initial_infected + initial_exposed, replace=False)
    for i, node in enumerate(seed_nodes):
        if i < initial_infected:
            state[node] = 2 # I
            rec_day = int(np.round(np.random.gamma(inf_mean, 1.0)))
            events.append((rec_day, 2, node))
        else:
            state[node] = 1 # E
            onset_day = int(np.round(np.random.gamma(inc_mean, 1.0)))
            events.append((onset_day, 1, node))
            
    max_sim_time = len(target_rt_array)
    daily_incidence = np.zeros(max_sim_time + 1)
    daily_incidence[0] = initial_infected
    
    true_rt_numerator = np.zeros(max_sim_time + 1)
    true_rt_denominator = np.zeros(max_sim_time + 1)
    
    events.sort(key=lambda x: x[0])
    
    for t in range(1, max_sim_time + 1):
        onset_cohort = []
        while events and events[0][0] <= t:
            ev_t, ev_type, node = events.pop(0)
            if ev_type == 1:
                if state[node] == 1:
                    state[node] = 2
                    daily_incidence[t] += 1
                    onset_cohort.append(node)
                    rec_day = t + int(np.round(np.random.gamma(inf_mean, 1.0)))
                    events.append((rec_day, 2, node))
            elif ev_type == 2:
                if state[node] == 2:
                    state[node] = 3
                    
        # Instead of fixed tau, we force exactly N*Rt using cohort pool to test if TRUE Rt differs from EpiEstim!
        if onset_cohort:
            target_rt = target_rt_array[t] if t < len(target_rt_array) else target_rt_array[-1]
            expected = len(onset_cohort) * target_rt
            target = int(np.floor(expected))
            if np.random.rand() < (expected - target): target += 1
            
            pool = []
            for node in onset_cohort:
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:
                        pool.append((node, neighbor))
            
            np.random.shuffle(pool)
            actual_infections = 0
            for source, target_node in pool:
                if state[target_node] == 0:
                    state[target_node] = 1
                    exposure_day = t + int(np.round(np.random.uniform(0, inf_mean)))
                    onset_day = exposure_day + int(np.round(np.random.gamma(inc_mean, 1.0)))
                    events.append((onset_day, 1, target_node))
                    actual_infections += 1
                    if actual_infections >= target: break
                        
            true_rt_numerator[t] += actual_infections
            true_rt_denominator[t] += len(onset_cohort)
            
        events.sort(key=lambda x: x[0])
        
    true_rt = np.zeros(max_sim_time + 1)
    for i in range(max_sim_time + 1):
        if true_rt_denominator[i] > 0:
            true_rt[i] = true_rt_numerator[i] / true_rt_denominator[i]
            
    return daily_incidence, true_rt

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

empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=1.0)
SHIFT_DAYS = 12
shifted_rt = np.zeros_like(empirical_rt)
if len(empirical_rt) > SHIFT_DAYS:
    shifted_rt[:-SHIFT_DAYS] = empirical_rt[SHIFT_DAYS:]
    shifted_rt[-SHIFT_DAYS:] = empirical_rt[-1]
rt_array = list(shifted_rt) + [shifted_rt[-1]] * 10

G = generate_network(10000, household_mean=5.2, community_mean=5.0, community_variance=25.0)

daily_incidence, true_rt = run_sim_track_rt(G, rt_array)
estimated_rt = estimate_rt_from_incidence(daily_incidence, prior_sd=1.0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rt_array[:len(true_rt)], label="Target Input Rt", linestyle=':', color='blue')
ax.plot(true_rt, label="True Underlying Rt", color='black')
ax.plot(estimated_rt, label="EpiEstim Estimated Rt", color='red')
ax.legend()
plt.savefig('../figures/test_true_rt.png')
