import numpy as np
import pandas as pd
from multiprocessing import Pool
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

def run_sim_sigmoidal(args):
    radius, eff, detect, k, d0, seed = args
    np.random.seed(seed)
    G = generate_network(2000)
    tau = 0.08
    res, deaths, vax = simulate_ring_vaccination(
        G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0,
        uptake=0.8, efficacy=eff, reporting_rate=detect, detection_delay=4.0, ring_radius=radius,
        sigmoidal_k=k, sigmoidal_d0=d0
    )
    if deaths < 0.005: return None
    return res, vax, deaths

if __name__ == '__main__':
    N_TRIALS = 300
    BASE_SEED = 20260629
    SIGMOIDAL_K = 0.5
    
    profiles = [
        ("Fast", SIGMOIDAL_K, 5.0),
        ("Intermediate", SIGMOIDAL_K, 10.0),
        ("Slow", SIGMOIDAL_K, 14.0)
    ]
    
    args_list = []
    seed = BASE_SEED
    for _, k, d0 in profiles:
        for r in [1, 2]:
            for _ in range(N_TRIALS):
                args_list.append((r, 0.4, 0.7, k, d0, seed))
                seed += 1
            
    print("Running Sigmoidal Scenarios...")
    with Pool() as p:
        results = p.map(run_sim_sigmoidal, args_list)
        
    parsed = {
        'fast_r1': [], 'fast_r2': [],
        'intermediate_r1': [], 'intermediate_r2': [],
        'slow_r1': [], 'slow_r2': []
    }
    idx = 0
    for _, key in zip(profiles, ['fast', 'intermediate', 'slow']):
        r1 = results[idx:idx+N_TRIALS]
        r1 = [x for x in r1 if x is not None]
        parsed[key + '_r1'] = [x[2]*100 for x in r1]
        idx += N_TRIALS
        
        r2 = results[idx:idx+N_TRIALS]
        r2 = [x for x in r2 if x is not None]
        parsed[key + '_r2'] = [x[2]*100 for x in r2]
        idx += N_TRIALS

    np.savez('data/fig5_sigmoidal.npz', 
             fast_r1=np.array(parsed['fast_r1'], dtype=object),
             fast_r2=np.array(parsed['fast_r2'], dtype=object),
             intermediate_r1=np.array(parsed['intermediate_r1'], dtype=object),
             intermediate_r2=np.array(parsed['intermediate_r2'], dtype=object),
             slow_r1=np.array(parsed['slow_r1'], dtype=object),
             slow_r2=np.array(parsed['slow_r2'], dtype=object),
             profile_names=np.array([p[0] for p in profiles]),
             profile_midpoints=np.array([p[2] for p in profiles]),
             profile_k=np.array([p[1] for p in profiles]))

    baseline = 45.4
    rows = []
    for label, key, midpoint in [
        ("Fast", "fast", 5),
        ("Intermediate", "intermediate", 10),
        ("Slow", "slow", 14),
    ]:
        for radius in [1, 2]:
            values = np.asarray(parsed[f"{key}_r{radius}"], dtype=float)
            deaths_averted = np.maximum(0, (baseline - values) / baseline * 100)
            rows.append({
                "profile": label,
                "immune_onset_midpoint_days": midpoint,
                "ring_radius": radius,
                "n_replicates": len(deaths_averted),
                "median_deaths_averted_pct": np.median(deaths_averted),
                "lower_95_ui": np.percentile(deaths_averted, 2.5),
                "upper_95_ui": np.percentile(deaths_averted, 97.5),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("data_and_results/immune_onset_sensitivity_summary.csv", index=False)
    with open("data_and_results/immune_onset_sensitivity_summary.md", "w") as f:
        f.write(summary.to_markdown(index=False, floatfmt=".1f"))
        f.write("\n")
    print("Saved Sigmoidal Data.")
