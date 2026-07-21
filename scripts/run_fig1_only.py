import numpy as np
import os
import networkx as nx
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
from concurrent.futures import ProcessPoolExecutor

print("Initializing Master Node: N=50,000 for Figure 1 only")
N = 50000
G = generate_network(N)
tau = calibrate_tau(G, target_R0=1.6, gamma=1.0/6.0, num_trials=20)
print(f"Master Tau Calibrated: {tau:.4f}")

def run_sim_fig1(args):
    daily_incidence = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.0, efficacy=0.0, initial_infected=5, return_time_series=True, max_sim_time=200
    )
    return daily_incidence

def main():
    os.makedirs('data', exist_ok=True)
    print("Generating Figure 1 Data (100 baseline runs)...")
    args_list = [{} for _ in range(100)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_sim_fig1, args_list))
    
    ts_all = np.array(results)
    np.save('data/fig1_timeseries.npy', ts_all)
    print(f"Saved fig1_timeseries.npy of shape {ts_all.shape}")

if __name__ == '__main__':
    main()
