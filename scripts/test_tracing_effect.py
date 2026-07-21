import json
import numpy as np
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination
from multiprocessing import Pool

POP_SIZE = 100000
N_REPS = 100
BASE_SEED = 20260630
HOUSEHOLD_MEAN = 5.0
COMMUNITY_MEAN = 5.0
COMMUNITY_VAR = 25.0
INCUBATION = 8.5
INFECTIOUS = 6.0
UPTAKE = 0.8
MAX_SIM_TIME = 90
SIGMOIDAL_D0 = 10.0

params = json.load(open('data_and_results/fitted_parameters.json'))
base_CFR = float(params.get('latest_adjusted_cfr', 0.454))
vax_CFR = base_CFR * 0.5
rt_array = json.load(open('data_and_results/rt_calibrated_tau_array.json'))['tau_array']

print("Generating network...")
GLOBAL_GRAPH = generate_network(POP_SIZE, household_mean=HOUSEHOLD_MEAN, community_mean=COMMUNITY_MEAN, community_variance=COMMUNITY_VAR)

def run_one(args):
    scenario, replicate, seed = args
    np.random.seed(seed)
    
    c, d, _ = simulate_ring_vaccination(
        GLOBAL_GRAPH,
        rt_array=rt_array,
        incubation_period=INCUBATION,
        infectious_period=INFECTIOUS,
        uptake=UPTAKE,
        efficacy=scenario['efficacy'],
        reporting_rate=scenario['reporting'],
        tracing_coverage=scenario['tracing'],
        detection_delay=4.0,
        ring_radius=scenario['radius'],
        max_sim_time=MAX_SIM_TIME,
        base_CFR=base_CFR,
        vax_CFR=vax_CFR,
        initial_infected=100,
        engine='cpp',
        seed=seed,
        sigmoidal_d0=SIGMOIDAL_D0
    )
    return {'scenario': scenario['name'], 'deaths': d * POP_SIZE}

def run_scenarios(scenarios):
    args = []
    for sc in scenarios:
        for rep in range(N_REPS):
            args.append((sc, rep, BASE_SEED + rep))
    
    with Pool(8) as pool:
        results = pool.map(run_one, args)
        
    df = {}
    for r in results:
        df.setdefault(r['scenario'], []).append(r['deaths'])
        
    for name, deaths in df.items():
        print(f"{name}: {np.median(deaths):.1f} median deaths")

scenarios = [
    {'name': '1. Base (30% Find, 30% Trace)', 'reporting': [0.3]*91, 'tracing': [0.3]*91, 'efficacy': 0.0, 'radius': 1},
    {'name': '2. High Find (70%), No Trace (0%)', 'reporting': np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76, 'tracing': [0.0]*91, 'efficacy': 0.0, 'radius': 1},
    {'name': '3. High Find (70%), High Trace (80%)', 'reporting': np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76, 'tracing': np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76, 'efficacy': 0.0, 'radius': 1},
    {'name': '4. High Find + No Trace + VAX', 'reporting': np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76, 'tracing': [0.0]*91, 'efficacy': 0.9, 'radius': 1},
    {'name': '5. High Find + High Trace + VAX', 'reporting': np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76, 'tracing': np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76, 'efficacy': 0.9, 'radius': 1},
]

if __name__ == '__main__':
    print("Running...")
    run_scenarios(scenarios)
