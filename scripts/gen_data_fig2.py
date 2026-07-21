import numpy as np
from multiprocessing import Pool
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

def run_sim_A(args):
    detect, eff = args
    G = generate_network(2000)
    tau = 0.08
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=eff, reporting_rate=detect, detection_delay=4.0, ring_radius=2,
        max_cases=None, max_daily_traces=1000
    )
    if deaths < 0.005: return None
    return deaths

def run_sim_B(args):
    drop_r2, add_delay_r2 = args
    G = generate_network(2000)
    tau = 0.08
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=0.4, reporting_rate=0.7, detection_delay=4.0, ring_radius=2,
        uptake_r2_drop=drop_r2, tracing_delay_r2_add=add_delay_r2,
        max_cases=None, max_daily_traces=1000
    )
    if deaths < 0.005: return None
    return deaths

if __name__ == '__main__':
    N_TRIALS = 30
    GRID_SIZE = 12
    
    # Scenario A: Case Detection vs Efficacy
    detections = np.linspace(0.1, 1.0, GRID_SIZE) # X-axis
    efficacies = np.linspace(0.2, 0.9, GRID_SIZE) # Y-axis
    
    args_A = []
    # Meshgrid populates with Y in outer loop, X in inner loop.
    for e in efficacies:
        for d in detections:
            for _ in range(N_TRIALS):
                args_A.append((d, e))
                
    # Scenario B: Radius 2 Relative Drop vs Radius 2 Added Delay
    added_delays_r2 = np.linspace(0.0, 14.0, GRID_SIZE) # X-axis
    drops_r2 = np.linspace(0.0, 1.0, GRID_SIZE) # Y-axis
    
    args_B = []
    for drop in drops_r2:
        for delay in added_delays_r2:
            for _ in range(N_TRIALS):
                args_B.append((drop, delay))
                
    print("Running Grid A (Detection vs Efficacy)...")
    with Pool() as p:
        res_A = p.map(run_sim_A, args_A)
        
    print("Running Grid B (R2 Drop vs R2 Delay)...")
    with Pool() as p:
        res_B = p.map(run_sim_B, args_B)
        
    def aggregate(res_list):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        idx = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                chunk = res_list[idx:idx+N_TRIALS]
                chunk = [x for x in chunk if x is not None]
                if chunk:
                    grid[i, j] = np.mean(chunk) * 100
                idx += N_TRIALS
        return grid
        
    grid_A = aggregate(res_A)
    grid_B = aggregate(res_B)
    
    np.save('data/fig2A_XX.npy', detections)
    np.save('data/fig2A_YY.npy', efficacies)
    np.save('data/fig2A_res.npy', grid_A)
    
    np.save('data/fig2B_XX.npy', added_delays_r2)
    np.save('data/fig2B_YY.npy', drops_r2)
    np.save('data/fig2B_res.npy', grid_B)
    
    print("Done generating Figure 2 contour data.")
