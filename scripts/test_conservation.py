import numpy as np
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

G = generate_network(1000, household_mean=5.2, community_mean=5.0, community_variance=5.0)

k = {
    'rt_array': rt_array, 'max_sim_time': max_sim_time,
    'initial_infected': 5, 'initial_exposed': 5,
    'baseline_tau': 0.1,
    'detection_delay': 4.0, 'reporting_rate': 0.0, 
    'uptake': 0.0, 
    'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42
}
res = simulate_ring_vaccination(G, **k)
inc = np.array(res['daily_incidence'])
num = np.array(res['true_rt_numerator'])
den = np.array(res['true_rt_denominator'])

print(f"Total True Rt Numerator (Exposures): {np.sum(num)}")
print(f"Total True Rt Denominator (Onsets): {np.sum(den)}")
print(f"Total Daily Incidence (Onsets): {np.sum(inc)}")
for i in range(15):
    print(f"Day {i:2d} | Exp: {num[i]:5.1f} | Onset: {den[i]:5.1f} | Inc: {inc[i]:5.1f}")
