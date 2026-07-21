import sys
sys.path.append('.')
from ebola_stochastic_ring import generate_network
from ebola_stochastic_ring_old import calibrate_tau
G = generate_network(1000, household_mean=5.2, community_mean=100.0, community_variance=100.0)
tau = calibrate_tau(G, 2.58, 1.0/6.0, num_trials=5)
print("Calibrated Tau:", tau)
