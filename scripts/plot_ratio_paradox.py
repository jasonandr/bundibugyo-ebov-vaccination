import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from ebola_stochastic_ring import generate_network
from ebola_stochastic_ring_old import calibrate_tau, simulate_ring_vaccination

def run_sim(var, reporting_rate):
    G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=var)
    tau = calibrate_tau(G, 2.58, 1.0/6.0, num_trials=5)
    
    rt_array = [2.58] * 65 # constant Rt forcing for clarity
    
    k = {
        'rt_array': rt_array, 'max_sim_time': 65,
        'initial_infected': 5, 'initial_exposed': 5,
        'baseline_tau': tau,
        'detection_delay': 4.0, 'reporting_rate': reporting_rate,
        'efficacy': 0.0, 'return_time_series': True, 'engine': 'cohort'
    }
    
    inc_reps = []
    num_reps = []
    den_reps = []
    for rep in range(50):
        k['seed'] = 42 + rep
        res = simulate_ring_vaccination(G, **k)
        inc_reps.append(res['daily_incidence'])
        num_reps.append(res['true_rt_numerator'])
        den_reps.append(res['true_rt_denominator'])
        
    return np.array(inc_reps), np.array(num_reps), np.array(den_reps)

def get_rts(num, den):
    rt_reps = np.zeros_like(num)
    rt_reps[:] = np.nan
    mask = den > 0
    rt_reps[mask] = num[mask] / den[mask]
    
    mean_of_ratios = np.nanmean(rt_reps, axis=0)
    
    sum_num = np.sum(num, axis=0)
    sum_den = np.sum(den, axis=0)
    ratio_of_means = np.zeros_like(sum_num)
    ratio_of_means[:] = np.nan
    m2 = sum_den > 0
    ratio_of_means[m2] = sum_num[m2] / sum_den[m2]
    
    return mean_of_ratios, ratio_of_means

# Unmitigated, High Var
inc1, num1, den1 = run_sim(160.0, 0.0)
mr1, rm1 = get_rts(num1, den1)

# Unmitigated, Low Var
inc2, num2, den2 = run_sim(5.0, 0.0)
mr2, rm2 = get_rts(num2, den2)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(np.mean(inc1, axis=0), label='Incidence (Var=160)', color='red')
axs[0].plot(np.mean(inc2, axis=0), label='Incidence (Var=5)', color='blue')
axs[0].set_title('Unmitigated Mean Incidence')
axs[0].legend()

axs[1].plot(mr1, label='Mean of Ratios (Var=160)', color='red', linestyle='--')
axs[1].plot(rm1, label='Ratio of Means (Var=160)', color='red', linestyle='-')
axs[1].plot(mr2, label='Mean of Ratios (Var=5)', color='blue', linestyle='--')
axs[1].plot(rm2, label='Ratio of Means (Var=5)', color='blue', linestyle='-')
axs[1].axhline(1.0, color='gray', linestyle=':')
axs[1].set_title('True Rt Calculation')
axs[1].legend()

plt.savefig('../figures/ratio_paradox.png')
