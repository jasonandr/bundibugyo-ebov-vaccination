import numpy as np
from multiprocessing import Pool
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

def run_sim(args):
    ve_d, ve_m = args
    G = generate_network(2000)
    tau = 0.08
    
    # Calculate vax_CFR floor based on VE_M|D
    base_cfr = 0.454
    vax_cfr_floor = base_cfr * (1.0 - ve_m)
    
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=ve_d, reporting_rate=0.7, detection_delay=4.0, ring_radius=2,
        base_CFR=base_cfr, vax_CFR=vax_cfr_floor,
        max_cases=None, max_daily_traces=1000
    )
    if deaths < 0.005: return None
    return deaths

if __name__ == '__main__':
    N_TRIALS = 30
    GRID_SIZE = 12
    
    ve_d_array = np.linspace(0.0, 1.0, GRID_SIZE)
    ve_m_array = np.linspace(0.0, 1.0, GRID_SIZE)
    
    args_list = []
    for ve_m in ve_m_array:
        for ve_d in ve_d_array:
            for _ in range(N_TRIALS):
                args_list.append((ve_d, ve_m))
                
    print("Running Grid for Figure 6...")
    with Pool() as p:
        res = p.map(run_sim, args_list)
        
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    idx = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            chunk = res[idx:idx+N_TRIALS]
            chunk = [x for x in chunk if x is not None]
            if chunk:
                grid[i, j] = np.mean(chunk) * 100
            idx += N_TRIALS
            
    np.save('data/fig6_XX.npy', ve_d_array)
    np.save('data/fig6_YY.npy', ve_m_array)
    np.save('data/fig6_res.npy', grid)
    
    print("Done generating Figure 6 contour data.")
