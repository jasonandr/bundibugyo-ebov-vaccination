import json
import numpy as np
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from multiprocessing import Pool

POP_SIZE = 50000
N_REPS = 100
BASE_SEED = 20260630

params = json.load(open('data_and_results/fitted_parameters.json'))
base_CFR = float(params.get('latest_adjusted_cfr', 0.454))
rt_array = json.load(open('data_and_results/rt_calibrated_tau_array.json'))['tau_array']
GLOBAL_GRAPH = generate_network(POP_SIZE, household_mean=5.0, community_mean=5.0, community_variance=25.0)

def run_one(args):
    efficacy, radius, engine, seed = args
    np.random.seed(seed)
    
    # Simple setup: Flat 30% operations so outbreak spreads, allowing vaccine to work
    c, d, _ = simulate_ring_vaccination(
        GLOBAL_GRAPH,
        rt_array=None,
        baseline_tau=0.08,
        incubation_period=8.5,
        infectious_period=6.0,
        uptake=0.8,
        efficacy=efficacy,
        reporting_rate=0.3,
        tracing_coverage=0.3,
        detection_delay=4.0,
        ring_radius=radius,
        max_sim_time=90,
        base_CFR=base_CFR,
        vax_CFR=base_CFR * 0.5,
        initial_infected=10,
        engine=engine,
        seed=seed,
        allow_pep=True,
        sigmoidal_d0=5.0
    )
    return {'scenario': f"{engine}_vax_{efficacy>0}", 'deaths': d * POP_SIZE}

scenarios = []
for rep in range(N_REPS):
    seed = BASE_SEED + rep
    scenarios.append((0.0, 1, 'python', seed))
    scenarios.append((0.9, 1, 'python', seed))
    scenarios.append((0.0, 1, 'cpp', seed))
    scenarios.append((0.9, 1, 'cpp', seed))

if __name__ == '__main__':
    with Pool(8) as pool:
        results = pool.map(run_one, scenarios)
        
    df = {}
    for r in results:
        df.setdefault(r['scenario'], []).append(r['deaths'])
        
    for name, deaths in df.items():
        print(f"{name}: {np.median(deaths):.1f} median deaths")
