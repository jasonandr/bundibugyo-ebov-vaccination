import numpy as np
from ebola_stochastic_ring import generate_network
import test_pooled_rt_network as test_pool
import test_deterministic_rt_network as test_det

def compare():
    rt_array = [2.5] * 100
    G = test_pool.generate_network(1000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    
    # Run deterministic (individual)
    np.random.seed(42)
    inc_det = test_det.run_deterministic_sim(G.copy(), rt_array, max_sim_time=40)
    
    # Run pooled
    np.random.seed(42)
    inc_pool = test_pool.run_pooled_sim(G.copy(), rt_array, max_sim_time=40)
    
    print("Det:", inc_det[:30])
    print("Pool:", inc_pool[:30])

compare()
