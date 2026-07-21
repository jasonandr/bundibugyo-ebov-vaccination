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

import ebola_stochastic_ring_old
original_func = ebola_stochastic_ring_old.simulate_ring_vaccination

def intercept_func(G, **kwargs):
    return original_func(G, **kwargs)

