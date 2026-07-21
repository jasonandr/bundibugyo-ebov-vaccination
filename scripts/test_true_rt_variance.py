import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network
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

empirical_rt = estimate_rt_from_incidence(cases_inc_empirical, prior_sd=1.0)
rt_array = np.array(list(empirical_rt) + [empirical_rt[-1]]*50)

def run_pooled_sim_binomial_track_rt(G, rt_array):
    N = len(G.nodes)
    state = np.zeros(N, dtype=int)
    events = []
    
    initial_infected = 5
    initial_exposed = 20
    inc_mean = 8.5
    inf_mean = 6.0
    
    seed_center = np.random.randint(0, N)
    pool = [seed_center]
    seed_nodes = set([seed_center])
    while len(seed_nodes) < (initial_infected + initial_exposed) and pool:
        curr = pool.pop(0)
        for neighbor in G.neighbors(curr):
            if neighbor not in seed_nodes:
                seed_nodes.add(neighbor)
                pool.append(neighbor)
                if len(seed_nodes) == (initial_infected + initial_exposed): break
    
    seed_nodes = list(seed_nodes)
    for i, node in enumerate(seed_nodes):
        if i < initial_infected:
            state[node] = 2
            events.append((int(np.round(np.random.gamma(inf_mean, 1.0))), 2, node))
        else:
            state[node] = 1
            events.append((int(np.round(np.random.gamma(inc_mean, 1.0))), 1, node))
            
    max_sim_time = len(rt_array)
    
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
                    onset_cohort.append(node)
                    events.append((t + int(np.round(np.random.gamma(inf_mean, 1.0))), 2, node))
            elif ev_type == 2:
                if state[node] == 2:
                    state[node] = 3
                    
        if onset_cohort:
            target_rt = rt_array[t] if t < len(rt_array) else rt_array[-1]
            expected = len(onset_cohort) * target_rt
            
            pool = []
            for node in onset_cohort:
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:
                        pool.append((node, neighbor))
            
            if len(pool) > 0:
                p = min(1.0, expected / len(pool))
                infections = np.random.binomial(len(pool), p)
                np.random.shuffle(pool)
                actual = 0
                for source, target_node in pool:
                    if state[target_node] == 0:
                        state[target_node] = 1
                        exp = t + int(np.round(np.random.uniform(0, inf_mean)))
                        events.append((exp + int(np.round(np.random.gamma(inc_mean, 1.0))), 1, target_node))
                        actual += 1
                        if actual >= infections: break
                        
                true_rt_numerator[t] += actual
                true_rt_denominator[t] += len(onset_cohort)
                        
        events.sort(key=lambda x: x[0])
        
    true_rt = np.zeros(max_sim_time + 1)
    true_rt[:] = np.nan
    for i in range(max_sim_time + 1):
        if true_rt_denominator[i] > 0:
            true_rt[i] = true_rt_numerator[i] / true_rt_denominator[i]
            
    return true_rt

G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=25.0)

results = []
for _ in range(50):
    res = run_pooled_sim_binomial_track_rt(G, rt_array)
    results.append(res)
    
fig, ax = plt.subplots(figsize=(12, 6))
for res in results:
    # only plot the part corresponding to empirical length
    ax.plot(res[:len(empirical_rt)], color='red', alpha=0.2)
ax.plot(empirical_rt, color='blue', linewidth=2.5, label="Empirical Rt Target")
ax.legend()
plt.ylim(0, 15)
plt.savefig('../figures/test_true_rt_variance.png')
