import numpy as np
from multiprocessing import Pool
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

def run_matrix_sim(args):
    ve_d, ve_m, base_cfr = args
    G = generate_network(2000)
    vax_cfr_floor = base_cfr * (1.0 - ve_m)
    
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=0.08, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=ve_d, reporting_rate=0.7, detection_delay=4.0, ring_radius=2,
        base_CFR=base_cfr, vax_CFR=vax_cfr_floor, max_cases=None, max_daily_traces=1000
    )
    if deaths < 0.005: return None
    return deaths

def run_violin_sim(args):
    ring_radius, detect_rate, base_cfr = args
    G = generate_network(5000)
    vax_cfr_floor = base_cfr * 0.50 # 50% relative reduction baseline
    
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=0.08, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=0.30, reporting_rate=detect_rate, detection_delay=4.0, ring_radius=ring_radius,
        base_CFR=base_cfr, vax_CFR=vax_cfr_floor, max_cases=None, max_daily_traces=100
    )
    if deaths < 0.005: return None
    return deaths, vax

if __name__ == '__main__':
    # 1. Mortality Matrices
    MATRIX_TRIALS = 30
    GRID_SIZE = 12
    ve_d_array = np.linspace(0.0, 1.0, GRID_SIZE)
    ve_m_array = np.linspace(0.0, 1.0, GRID_SIZE)
    
    for year, b_cfr in [("2007", 0.25), ("2012", 0.51)]:
        print(f"Running Mortality Matrix for {year}...")
        args_list = []
        for ve_m in ve_m_array:
            for ve_d in ve_d_array:
                for _ in range(MATRIX_TRIALS):
                    args_list.append((ve_d, ve_m, b_cfr))
                    
        with Pool() as p:
            res = p.map(run_matrix_sim, args_list)
            
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        idx = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                chunk = res[idx:idx+MATRIX_TRIALS]
                chunk = [x for x in chunk if x is not None]
                if chunk:
                    grid[i, j] = np.mean(chunk) * 100
                idx += MATRIX_TRIALS
        np.save(f'data/supp_matrix_{year}_res.npy', grid)

    # 2. Violin Plots
    VIOLIN_TRIALS = 200 # More trials for violin plot density
    detect_rates = [0.2, 0.4, 0.6]
    for year, b_cfr in [("2007", 0.25), ("2012", 0.51)]:
        print(f"Running Violins for {year}...")
        args_list = []
        for r in [1, 2]:
            for d in detect_rates:
                for _ in range(VIOLIN_TRIALS):
                    args_list.append((r, d, b_cfr))
                    
        with Pool() as p:
            res = p.map(run_violin_sim, args_list)
            
        import pickle
        with open(f'data/supp_violin_{year}_res.pkl', 'wb') as f:
            pickle.dump(res, f)
            
    print("Supplemental Data Generation Complete.")
