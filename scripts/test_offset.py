import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from ebola_stochastic_ring_old import calibrate_tau

with open("../data_and_results/fitted_parameters.json", "r") as f:
    params = json.load(f)
rt_array = params.get("Rt_array", [])

# Let's use the exact Rt array WITHOUT the shifted hack for the engine, 
# because the engine t=0 will now literally mean May 2!
SHIFT_DAYS = 12

# Pad the end of rt_array so it doesn't run out
rt_array_padded = list(rt_array) + [rt_array[-1]] * 60

print("Calibrating tau...")
G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=160.0)
R_max = max(rt_array_padded)
baseline_tau = calibrate_tau(G, R_max, 1.0/6.0, num_trials=5)

print("Simulating...")
inc = simulate_ring_vaccination(
    G, rt_array=rt_array_padded, baseline_tau=baseline_tau,
    incubation_period=8.5, infectious_period=6.0,
    uptake=0.0, efficacy=0.0, reporting_rate=0.0,
    detection_delay=4.0, tracing_delay=0.0, ring_radius=1,
    initial_infected=1, initial_exposed=0,
    max_sim_time=len(rt_array) + SHIFT_DAYS, return_time_series=True, engine='cpp'
)

print(f"Cases at day 12 (May 14): {np.sum(inc[:13])}")
print(f"Total cases at end: {np.sum(inc)}")
