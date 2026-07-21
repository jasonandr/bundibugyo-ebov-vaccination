import sys
sys.path.append('.')
from ebola_stochastic_ring_old import simulate_ring_vaccination
from ebola_stochastic_ring import generate_network
import numpy as np
import json

with open('../data_and_results/fitted_parameters.json', 'r') as f:
    params = json.load(f)
rt_array = np.array(params.get('Rt_array', []))
max_sim_time = 100
if len(rt_array) < max_sim_time:
    rt_array = list(rt_array) + [rt_array[-1]] * (max_sim_time - len(rt_array))

G = generate_network(100000, household_mean=5.2, community_mean=100.0, community_variance=100.0)

# I will count total infections generated in engine cohort vs branching
def run_comparison():
    inc_reps = []
    for rep in range(100):
        k = {
            'rt_array': rt_array, 'max_sim_time': max_sim_time,
            'initial_infected': 5, 'initial_exposed': 5,
            'baseline_tau': 0.1,
            'detection_delay': 4.0, 'reporting_rate': 0.0, 
            'uptake': 0.0, 
            'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42 + rep
        }
        res = simulate_ring_vaccination(G, **k)
        inc_reps.append(res['daily_incidence'])

    inc_reps = np.array(inc_reps)
    print("Simulated Engine Cohort:")
    print("Day 30 Mean:", np.mean(inc_reps[:, 30]))
    print("Day 60 Mean:", np.mean(inc_reps[:, 60]))

run_comparison()
