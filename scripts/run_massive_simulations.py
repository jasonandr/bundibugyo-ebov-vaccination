import numpy as np
import os
import networkx as nx
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
from concurrent.futures import ProcessPoolExecutor

print("Initializing Master Node: N=10,000 for Major Outbreaks")
N = 10000
G = generate_network(N)
tau = calibrate_tau(G, target_R0=1.6, gamma=1.0/6.0, num_trials=30)
print(f"Master Tau Calibrated: {tau:.4f}")

def run_major_sim(args):
    """Runs simulations until a MAJOR outbreak occurs (e.g. >50 cases)."""
    sim_kwargs = {k: v for k, v in args.items() if k not in ['i', 'j', 'b', 'r', 'name', 'eff']}
    
    max_retries = 20
    for _ in range(max_retries):
        res = simulate_ring_vaccination(
            G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
            **sim_kwargs
        )
        total_cases = res[0] * N
        if total_cases >= 50:
            if 'return_time_series' in sim_kwargs and sim_kwargs['return_time_series']:
                return res[0], res[1], res[2], res[3]
            return res[0], res[1], res[2]
            
    # If it completely fails 20 times to establish, return the last one anyway to prevent infinite loop.
    if 'return_time_series' in sim_kwargs and sim_kwargs['return_time_series']:
        return res[0], res[1], res[2], res[3]
    return res[0], res[1], res[2]

def main():
    os.makedirs('data', exist_ok=True)
    
    # 1. Figure 2 Data
    print("Generating Figure 2 Data (Conditioned on Major Outbreaks)...")
    grid_size = 5
    num_trials = 50
    # Panel A
    efficacies = np.linspace(0.2, 1.0, grid_size)
    coverages = np.linspace(0.2, 1.0, grid_size)
    EE, CC = np.meshgrid(efficacies, coverages)
    args_list_2A = []
    for i in range(grid_size):
        for j in range(grid_size):
            for _ in range(num_trials):
                args_list_2A.append({'uptake':0.0, 'efficacy':EE[i,j], 'tracing_coverage':CC[i,j], 'vaccine_acceptability':0.9, 'detection_delay':4.0, 'i':i, 'j':j})
    
    res_2A = np.zeros((grid_size, grid_size))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_major_sim, args_list_2A))
    for args, r in zip(args_list_2A, results):
        res_2A[args['i'], args['j']] += r[1] * 100 / num_trials # Average deaths percentage
    
    np.save('data/fig2A_EE.npy', EE)
    np.save('data/fig2A_CC.npy', CC)
    np.save('data/fig2A_res.npy', res_2A)
    
    # Panel B
    delays = np.linspace(2.0, 10.0, grid_size)
    coverages_B = np.linspace(0.2, 1.0, grid_size)
    DD, CC_B = np.meshgrid(delays, coverages_B)
    args_list_2B = []
    for i in range(grid_size):
        for j in range(grid_size):
            for _ in range(num_trials):
                args_list_2B.append({'uptake':0.0, 'efficacy':0.7, 'tracing_coverage':CC_B[i,j], 'vaccine_acceptability':0.9, 'detection_delay':DD[i,j], 'i':i, 'j':j})
                
    res_2B = np.zeros((grid_size, grid_size))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_major_sim, args_list_2B))
    for args, r in zip(args_list_2B, results):
        res_2B[args['i'], args['j']] += r[1] * 100 / num_trials
        
    np.save('data/fig2B_DD.npy', DD)
    np.save('data/fig2B_CC.npy', CC_B)
    np.save('data/fig2B_res.npy', res_2B)

    # 3. Figure 3 Data
    print("Generating Figure 3 Data...")
    bandwidths = np.linspace(20, 250, 12)
    num_trials_3 = 50
    args_3 = []
    for b in bandwidths:
        for _ in range(num_trials_3):
            args_3.append({'uptake':0.0, 'efficacy':0.8, 'tracing_coverage':0.8, 'vaccine_acceptability':0.9, 'ring_radius':1, 'max_daily_traces':int(b), 'b':b, 'r':1})
            args_3.append({'uptake':0.0, 'efficacy':0.8, 'tracing_coverage':0.8, 'vaccine_acceptability':0.9, 'ring_radius':2, 'max_daily_traces':int(b), 'b':b, 'r':2})
            
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_major_sim, args_3))
        
    r1_cases, r2_cases, r1_vax, r2_vax = {}, {}, {}, {}
    for args, r in zip(args_3, results):
        b = args['b']
        rad = args['r']
        if rad == 1:
            r1_cases[b] = r1_cases.get(b, 0) + (r[0]*100)/num_trials_3
            r1_vax[b] = r1_vax.get(b, 0) + r[2]/num_trials_3
        else:
            r2_cases[b] = r2_cases.get(b, 0) + (r[0]*100)/num_trials_3
            r2_vax[b] = r2_vax.get(b, 0) + r[2]/num_trials_3
            
    r1c = [r1_cases[b] for b in bandwidths]
    r2c = [r2_cases[b] for b in bandwidths]
    r1v = [r1_vax[b] for b in bandwidths]
    r2v = [r2_vax[b] for b in bandwidths]
    
    np.savez('data/fig3_data.npz', bandwidths=bandwidths, r1_cases=r1c, r2_cases=r2c, r1_vax=r1v, r2_vax=r2v)
    
    # 4. Figure 4 Data
    print("Generating Figure 4 Data...")
    grid_size_4 = 6
    num_trials_4 = 30
    multipliers = np.linspace(1.0, 3.0, grid_size_4)
    efficacies_4 = np.linspace(0.1, 1.0, grid_size_4)
    MM, EE_4 = np.meshgrid(multipliers, efficacies_4)
    args_4A = []
    for i in range(grid_size_4):
        for j in range(grid_size_4):
            for _ in range(num_trials_4):
                args_4A.append({'uptake':0.8, 'efficacy':EE_4[i,j], 'risk_compensation_multiplier':MM[i,j], 'trust_uptake_dependency':False, 'i':i, 'j':j})
                
    args_base = [{'uptake':0.0, 'efficacy':0.0, 'tracing_coverage':0.0, 'vaccine_acceptability':0.0} for _ in range(50)]
    with ProcessPoolExecutor() as executor:
        base_results = list(executor.map(run_major_sim, args_base))
        res_4A_raw = list(executor.map(run_major_sim, args_4A))
        
    baseline_deaths = np.mean([r[1]*100 for r in base_results])
    res_4A = np.zeros((grid_size_4, grid_size_4))
    for args, r in zip(args_4A, res_4A_raw):
        res_4A[args['i'], args['j']] += (r[1]*100 - baseline_deaths) / num_trials_4
        
    np.save('data/fig4A_MM.npy', MM)
    np.save('data/fig4A_EE.npy', EE_4)
    np.save('data/fig4A_res.npy', res_4A)
    np.save('data/baseline_deaths.npy', np.array([baseline_deaths]))
    
    # Panel B
    line_effs = np.linspace(0.1, 1.0, 8)
    args_4B = []
    scenarios = [
        ("Base", 1.0, False),
        ("Trust", 1.0, True),
        ("Risk", 2.0, False),
        ("Both", 2.0, True)
    ]
    for eff in line_effs:
        for name, rc, tl in scenarios:
            for _ in range(num_trials_4):
                args_4B.append({'uptake':0.8, 'efficacy':eff, 'risk_compensation_multiplier':rc, 'trust_uptake_dependency':tl, 'name':name, 'eff':eff})
                
    with ProcessPoolExecutor() as executor:
        res_4B_raw = list(executor.map(run_major_sim, args_4B))
        
    results_4B = {s[0]: {} for s in scenarios}
    for args, r in zip(args_4B, res_4B_raw):
        results_4B[args['name']][args['eff']] = results_4B[args['name']].get(args['eff'], 0) + (r[1]*100)/num_trials_4
        
    res_dict = {n: [results_4B[n][e] for e in line_effs] for n,_,_ in scenarios}
    np.savez('data/fig4B_data.npz', line_effs=line_effs, **res_dict)
    
    # 5. Figure 5 Data (LHS)
    print("Generating Figure 5 Data (LHS)...")
    num_samples = 400
    num_trials_5 = 3
    X = np.zeros((num_samples, 6))
    bounds = [
        (0.2, 1.0), # reporting
        (0.2, 1.0), # tracing_cov
        (0.4, 1.0), # vax_acc
        (2.0, 8.0), # delay
        (0.4, 0.75), # base_cfr
        (0.15, 0.35) # vax_cfr
    ]
    for i, (low, high) in enumerate(bounds):
        X[:, i] = np.random.uniform(low, high, num_samples)
        
    args_5 = []
    for i in range(num_samples):
        for _ in range(num_trials_5):
            args_5.append({
                'uptake':0.0, 'efficacy':0.7,
                'reporting_rate':X[i,0], 'tracing_coverage':X[i,1], 'vaccine_acceptability':X[i,2],
                'detection_delay':X[i,3], 'base_CFR':X[i,4], 'vax_CFR':X[i,5],
                'i': i
            })
        
    with ProcessPoolExecutor() as executor:
        res_5_raw = list(executor.map(run_major_sim, args_5))
        
    results_deaths_5 = np.zeros(num_samples)
    for args, r in zip(args_5, res_5_raw):
        results_deaths_5[args['i']] += (r[1]*100)/num_trials_5

    np.save('data/fig5_X.npy', X)
    np.save('data/fig5_deaths.npy', results_deaths_5)
    
    print("ALL MASSIVE SIMULATIONS COMPLETE AND PERSISTED TO DISK.")

if __name__ == '__main__':
    main()
