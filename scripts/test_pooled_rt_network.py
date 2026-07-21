import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network
from current_outbreak_data import cumulative_confirmed_cases
from plot_rt_calibration_test import estimate_rt_from_incidence, fast_rt
import time

def run_pooled_sim(G, rt_array, initial_infected=25, initial_exposed=0, max_sim_time=100, inc_mean=8.5, inf_mean=6.0):
    N = len(G.nodes)
    state = np.zeros(N, dtype=int)
    events = []
    
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
            
    daily_incidence = np.zeros(max_sim_time + 1)
    daily_incidence[0] = initial_infected
    
    if onset_cohort_day0:
        target_rt = rt_array[0] if 0 < len(rt_array) else 0
        expected = len(onset_cohort_day0) * target_rt
        target = int(np.floor(expected))
        if np.random.rand() < (expected - target): target += 1
            
        pool = []
        for node in onset_cohort_day0:
            for neighbor in G.neighbors(node):
                if state[neighbor] == 0:
                    pool.append((node, neighbor))
                    
        np.random.shuffle(pool)
        actual_infections = 0
        for source, target_node in pool:
            if state[target_node] == 0:
                state[target_node] = 1
                exposure_day = int(np.round(np.random.uniform(0, inf_mean)))
                onset_day = exposure_day + int(np.round(np.random.gamma(inc_mean, 1.0)))
                events.append((onset_day, 1, target_node))
                actual_infections += 1
                if actual_infections >= target: break

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
                        
        events.sort(key=lambda x: x[0])
        
    return daily_incidence

def main():
    print("Loading data...")
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
    
    print("Generating network...")
    G = generate_network(30000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    
    n_runs = 100
    print(f"Running {n_runs} POOLED deterministic simulations...")
    all_inc = []
    max_t = len(cases_inc_empirical) - 1
    
    init_inf = 5
    init_exp = 20
    
    for i in range(n_runs):
        inc = run_pooled_sim(G, rt_array, initial_infected=init_inf, initial_exposed=init_exp, max_sim_time=max_t)
        all_inc.append(inc[:max_t+1])
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1} runs...")
            
    all_inc = np.array(all_inc)
    mean_sim_inc = np.mean(all_inc, axis=0)
    
    print("Estimating Rt...")
    simulated_rt_mean = estimate_rt_from_incidence(mean_sim_inc, prior_sd=1.0)
    spaghetti_rt = fast_rt(all_inc)
    
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.bar(df_daily['Date'][:max_t+1], cases_inc_empirical, alpha=0.3, color='blue', label='Empirical Incidence (Daily)')
    ax1.plot(df_daily['Date'][:max_t+1], mean_sim_inc, color='red', linewidth=2, label='Mean Simulated Incidence')
    for inc in all_inc:
        ax1.plot(df_daily['Date'][:max_t+1], inc, color='red', alpha=0.05, linewidth=1)
    
    ax1.set_ylabel('Daily New Cases')
    ax1.set_title('A: Daily Incidence (Pooled Cohort Edge Targeting - SCALED VOLUME)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_daily['Date'][:max_t+1], empirical_rt, color='blue', linewidth=2, label='Empirical Rt (Target)')
    ax2.plot(df_daily['Date'][:max_t+1], simulated_rt_mean, color='red', linewidth=2, label='Mean Simulated Rt (from Mean Incidence)')
    
    for rt_traj in spaghetti_rt:
        ax2.plot(df_daily['Date'][:max_t+1], rt_traj[:max_t+1], color='red', alpha=0.05, linewidth=1)
        
    ax2.plot(df_daily['Date'][:max_t+1], shifted_rt[:max_t+1], color='blue', linestyle='--', alpha=0.5, label='Shifted Forcing Rt')
    
    ax2.set_ylabel('Effective Reproduction Number (Rt)')
    ax2.set_title('B: Effective Reproduction Number (Pooled Cohort Edge Targeting - 100 Runs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = f"../figures/rt_pooled_SCALED_100_runs.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
