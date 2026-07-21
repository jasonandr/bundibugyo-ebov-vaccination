import sys
sys.path.append('.')
from ebola_stochastic_ring_old import simulate_ring_vaccination
from ebola_stochastic_ring import generate_network
import numpy as np

G = generate_network(100000, household_mean=5.2, community_mean=100.0, community_variance=100.0)

# Overwrite function to print
import ebola_stochastic_ring_old
original_func = ebola_stochastic_ring_old.simulate_ring_vaccination

def intercept_func(G, **kwargs):
    # just run standard, we will print inside
    return original_func(G, **kwargs)

# We will just patch the file directly
