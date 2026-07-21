import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('.')
from ebola_stochastic_ring import generate_network
from ebola_stochastic_ring_old import calibrate_tau, simulate_ring_vaccination

def main():
    with open('../data_and_results/fitted_parameters.json', 'r') as f:
        params = json.load(f)
    rt_array = params.get('Rt_array', [])
    
    max_sim_time = 65
    if len(rt_array) < max_sim_time:
        rt_array = list(rt_array) + [rt_array[-1]] * (max_sim_time - len(rt_array))
        
    fig, ax = plt.subplots(figsize=(10,6))

    variances = [160.0, 20.0, 5.0]
    colors = ['red', 'green', 'blue']
    
    for var, color in zip(variances, colors):
        print(f"Running variance {var}...")
        G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=var)
        tau = calibrate_tau(G, 2.58, 1.0/6.0, num_trials=5)
        
        reps_unmitigated = []
        reps_mitigated = []
        
        for rep in range(10): # average over 10 replicates for smoother curves
            k_base = {
                'rt_array': rt_array, 'max_sim_time': max_sim_time,
                'initial_infected': 5, 'initial_exposed': 5,
                'baseline_tau': tau,
                'detection_delay': 4.0, 'reporting_rate': 0.8,
                'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort', 'seed': 42 + rep
            }
            res_mit = simulate_ring_vaccination(G, **k_base)
            reps_mitigated.append(res_mit['daily_incidence'])
            
            k_unmit = k_base.copy()
            k_unmit['reporting_rate'] = 0.0
            res_unmit = simulate_ring_vaccination(G, **k_unmit)
            reps_unmitigated.append(res_unmit['daily_incidence'])
            
        mean_unmit = np.mean(reps_unmitigated, axis=0)
        mean_mit = np.mean(reps_mitigated, axis=0)
        
        ax.plot(mean_unmit, label=f'Unmitigated (Var={var})', linestyle='--', color=color, alpha=0.8)
        ax.plot(mean_mit, label=f'80% Tracing (Var={var})', linestyle='-', color=color, alpha=0.8)

    ax.set_title("Simulated Incidence vs Community Variance (Seed=5, 10 Reps)")
    ax.set_xlabel("Days since May 15")
    ax.set_ylabel("Daily Incidence")
    ax.legend()
    plt.savefig('../figures/test_shape_variance.png')
    print("Done plotting.")
    
if __name__ == '__main__':
    main()
