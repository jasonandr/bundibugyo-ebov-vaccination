import numpy as np
import json
import os
import sys

from test_base_case_500 import generate_network
from ebola_stochastic_ring import simulate_ring_vaccination
from paths import result_path

def run_scenarios():
    N_POP = 100000
    G = generate_network(N_POP)
    
    # 2-week linear ramps (Day 0 to Day 14)
    reporting_ramp = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
    tracing_ramp = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76
    
    BASELINE_TAU = 0.08
    I0 = 5
    E0 = 0
    with open(result_path('rt_calibrated_tau_array.json')) as f:
        tau_array = json.load(f)['tau_array']

    for seed in range(1, 11):
        # Base case: No vaccine, 30% reporting, 30% tracing
        res_base = simulate_ring_vaccination(
            G, rt_array=tau_array, baseline_tau=BASELINE_TAU, ring_radius=0, 
            efficacy=0.0, reporting_rate=[0.3]*91, detection_delay=4.0, max_sim_time=90, 
            initial_infected=I0, initial_exposed=E0, engine='cpp', seed=seed,
            tracing_coverage=[0.3]*91
        )
        
        # Enhanced case finding/contact tracing: No vaccine, ramping
        res_enh = simulate_ring_vaccination(
            G, rt_array=tau_array, baseline_tau=BASELINE_TAU, ring_radius=0, 
            efficacy=0.0, reporting_rate=reporting_ramp, detection_delay=4.0, max_sim_time=90, 
            initial_infected=I0, initial_exposed=E0, engine='cpp', seed=seed,
            tracing_coverage=tracing_ramp
        )
        
        # Enhanced + Vax
        res_vax = simulate_ring_vaccination(
            G, rt_array=tau_array, baseline_tau=BASELINE_TAU, ring_radius=1, 
            efficacy=1.0, reporting_rate=reporting_ramp, detection_delay=4.0, max_sim_time=90, 
            initial_infected=I0, initial_exposed=E0, engine='cpp', seed=seed,
            tracing_coverage=tracing_ramp, uptake=1.0, vaccine_acceptability=1.0, allow_pep=True
        )
        print(f"Seed {seed}: Base {res_base[1]*N_POP:.0f} | Enh {res_enh[1]*N_POP:.0f} | Vax {res_vax[1]*N_POP:.0f}")

if __name__ == '__main__':
    run_scenarios()
