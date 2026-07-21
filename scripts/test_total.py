import sys
sys.path.append('.')
from ebola_stochastic_ring_old import simulate_ring_vaccination
from ebola_stochastic_ring import generate_network
import numpy as np

G = generate_network(10000, household_mean=5.2, community_mean=100.0, community_variance=100.0)

rt_array = [2.0] * 100
total = []
for rep in range(100):
    k = {
        'rt_array': rt_array, 'max_sim_time': 100,
        'initial_infected': 5, 'initial_exposed': 5,
        'baseline_tau': 0.1,
        'detection_delay': 4.0, 'reporting_rate': 0.0, 
        'uptake': 0.0, 
        'efficacy': 0.0, 'engine': 'cohort', 'seed': 42 + rep, 'return_time_series': True
    }
    res = simulate_ring_vaccination(G, **k)
    total.append(sum(res['daily_incidence']))

print("Mean total infected:", np.mean(total))
