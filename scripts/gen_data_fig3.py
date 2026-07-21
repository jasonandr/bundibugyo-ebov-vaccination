import numpy as np
from multiprocessing import Pool
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

BASE_SEED = 20260628


def run_sim(args):
    radius, eff, detect, max_vax, seed = args
    np.random.seed(seed)
    G = generate_network(2000)
    tau = 0.08
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=eff, reporting_rate=detect, detection_delay=4.0, ring_radius=radius,
        max_cases=None, max_daily_traces=1000,
        max_vaccines=max_vax if max_vax is not None else None
    )
    return res, vax, deaths

def run_sim_with_cap(args):
    radius, eff, detect, max_vax, seed = args
    np.random.seed(seed)
    G = generate_network(2000)
    tau = 0.08
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=eff, reporting_rate=detect, detection_delay=4.0, ring_radius=radius,
        max_cases=None, max_daily_traces=1000,
        max_vaccines=max_vax
    )
    return res, vax, deaths

if __name__ == '__main__':
    N_TRIALS = 200
    
    # Scenario A: Efficacy
    efficacies = [0.2, 0.4, 0.6]
    res_A_r1 = []
    res_A_r2 = []
    
    # Scenario B: Detection
    detections = [0.4, 0.6, 0.8]
    res_B_r1 = []
    res_B_r2 = []
    
    # Scenario C: Stockpile
    caps = [500, 1000, 3000]
    res_C_r1 = []
    res_C_r2 = []
    
    seed_counter = BASE_SEED

    def next_seed():
        nonlocal_seed[0] += 1
        return nonlocal_seed[0]

    nonlocal_seed = [seed_counter]

    args_A = []
    for e in efficacies:
        for r in [1, 2]:
            for _ in range(N_TRIALS):
                args_A.append((r, e, 0.7, 99999, next_seed()))
                
    args_B = []
    for d in detections:
        for r in [1, 2]:
            for _ in range(N_TRIALS):
                args_B.append((r, 0.4, d, 99999, next_seed()))
                
    args_C = []
    for c in caps:
        for r in [1, 2]:
            for _ in range(N_TRIALS):
                args_C.append((r, 0.4, 0.7, c, next_seed()))
                
    print("Running Scenario A...")
    with Pool() as p:
        results_A = p.map(run_sim, args_A)
    print("Running Scenario B...")
    with Pool() as p:
        results_B = p.map(run_sim, args_B)
    print("Running Scenario C...")
    with Pool() as p:
        results_C = p.map(run_sim_with_cap, args_C)
        
    def parse(res_list, n_tiers):
        parsed = {'r1_cases': [], 'r1_vax': [], 'r1_deaths': [], 'r2_cases': [], 'r2_vax': [], 'r2_deaths': []}
        idx = 0
        for tier in range(n_tiers):
            r1 = res_list[idx:idx+N_TRIALS]
            parsed['r1_cases'].append([x[0]*100 for x in r1])
            parsed['r1_vax'].append([x[1] for x in r1])
            parsed['r1_deaths'].append([x[2]*100 for x in r1])
            idx += N_TRIALS
            
            r2 = res_list[idx:idx+N_TRIALS]
            parsed['r2_cases'].append([x[0]*100 for x in r2])
            parsed['r2_vax'].append([x[1] for x in r2])
            parsed['r2_deaths'].append([x[2]*100 for x in r2])
            idx += N_TRIALS
        return parsed
        
    data_A = parse(results_A, len(efficacies))
    data_B = parse(results_B, len(detections))
    data_C = parse(results_C, len(caps))
    
    np.savez('data/fig3_violin_data.npz', 
             A_r1_cases=np.array(data_A['r1_cases'], dtype=object), A_r1_vax=np.array(data_A['r1_vax'], dtype=object), A_r1_deaths=np.array(data_A['r1_deaths'], dtype=object),
             A_r2_cases=np.array(data_A['r2_cases'], dtype=object), A_r2_vax=np.array(data_A['r2_vax'], dtype=object), A_r2_deaths=np.array(data_A['r2_deaths'], dtype=object),
             B_r1_cases=np.array(data_B['r1_cases'], dtype=object), B_r1_vax=np.array(data_B['r1_vax'], dtype=object), B_r1_deaths=np.array(data_B['r1_deaths'], dtype=object),
             B_r2_cases=np.array(data_B['r2_cases'], dtype=object), B_r2_vax=np.array(data_B['r2_vax'], dtype=object), B_r2_deaths=np.array(data_B['r2_deaths'], dtype=object),
             C_r1_cases=np.array(data_C['r1_cases'], dtype=object), C_r1_vax=np.array(data_C['r1_vax'], dtype=object), C_r1_deaths=np.array(data_C['r1_deaths'], dtype=object),
             C_r2_cases=np.array(data_C['r2_cases'], dtype=object), C_r2_vax=np.array(data_C['r2_vax'], dtype=object), C_r2_deaths=np.array(data_C['r2_deaths'], dtype=object),
             efficacies=efficacies, detections=detections, caps=caps)
    print("Saved Figure 3 Violin Data.")
