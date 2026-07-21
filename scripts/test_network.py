import numpy as np
import pandas as pd
import json
import sys
sys.path.append('.')
from ebola_stochastic_ring import generate_network
from ebola_stochastic_ring_old import calibrate_tau, simulate_ring_vaccination

with open('../data_and_results/fitted_parameters.json', 'r') as f:
    params = json.load(f)
rt_array = np.array(params.get('Rt_array', []))
max_sim_time = 100
if len(rt_array) < max_sim_time:
    rt_array = list(rt_array) + [rt_array[-1]] * (max_sim_time - len(rt_array))

# Huge community to prevent network saturation double-counting
G = generate_network(100000, household_mean=5.2, community_mean=100.0, community_variance=100.0)
tau = calibrate_tau(G, 2.58, 1.0/6.0, num_trials=5)

inc_reps = []
for rep in range(50):
    k = {
        'rt_array': rt_array, 'max_sim_time': max_sim_time,
        'initial_infected': 5, 'initial_exposed': 5,
        'baseline_tau': tau,
        'detection_delay': 4.0, 'reporting_rate': 0.0, 
        'uptake': 0.0, 
        'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42 + rep
    }
    res = simulate_ring_vaccination(G, **k)
    inc_reps.append(res['daily_incidence'])

inc_reps = np.array(inc_reps)
mean_inc = np.mean(inc_reps, axis=0)

print("Day | Mean Inc (K=100)")
for day in range(0, 80, 5):
    print(f"{day:3d} | {mean_inc[day]:8.2f}")
