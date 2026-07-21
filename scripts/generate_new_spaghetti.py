import os
import sys
import numpy as np
import ebola_stochastic_ring as sim
from paths import result_path
import json

def generate_spaghetti_data():
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    reps_per_job = 1000
    _G = sim.generate_network(10000) # smaller network just to generate trajectory examples, or wait, we need N=100000 for consistency!
    # Wait, earlier spaghetti used N=10000 because N=100000 takes too much memory for full time series tracking.
    # The actual code used N=10000. Let's use N=100000 for consistency with the rest of the paper!
    _G = sim.generate_network(100000)
    
    with open(result_path("fitted_parameters.json")) as f:
        params = json.load(f)
    
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = json.load(f)["tau_array"]
    max_tau = max(TAU_ARRAY)
    RT_ARRAY = [t * (6.066 / max_tau) for t in TAU_ARRAY]
    
    def make_ramp(target, duration=15, max_time=91):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)
        
    # Baseline ops
    base_rep = make_ramp(0.5)
    base_tr = make_ramp(0.5)
    
    # Enhanced ops
    enh_rep = make_ramp(0.7)
    enh_tr = make_ramp(0.8)
    
    BASE_VE = 0.45

    results = {
        "enh_no_vax": [],
        "enh_ring": [],
        "base_no_vax": [],
        "base_comm40": [],
        "base_hybrid": []
    }
    
    for i in range(reps_per_job):
        seed = array_id * 100000 + i
        
        # Panel A
        enh_nv = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.0, reporting_rate=enh_rep, tracing_coverage=enh_tr,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        enh_ring = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=BASE_VE, reporting_rate=enh_rep, tracing_coverage=enh_tr,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        # Panel B & C
        base_nv = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.0, reporting_rate=base_rep, tracing_coverage=base_tr,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        base_comm40 = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=1, max_vaccines=0,
            vaccine_effect=BASE_VE, reporting_rate=base_rep, tracing_coverage=base_tr,
            community_vax_coverage=0.4, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        base_hybrid = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=1, 
            vaccine_effect=BASE_VE, reporting_rate=base_rep, tracing_coverage=base_tr,
            community_vax_coverage=0.4, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        results["enh_no_vax"].append(np.array(enh_nv))
        results["enh_ring"].append(np.array(enh_ring))
        results["base_no_vax"].append(np.array(base_nv))
        results["base_comm40"].append(np.array(base_comm40))
        results["base_hybrid"].append(np.array(base_hybrid))

    out_dir = "data_and_results/new_spaghetti_chunks"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        f"{out_dir}/chunk_{array_id}.npz",
        enh_no_vax=np.array(results["enh_no_vax"], dtype=object),
        enh_ring=np.array(results["enh_ring"], dtype=object),
        base_no_vax=np.array(results["base_no_vax"], dtype=object),
        base_comm40=np.array(results["base_comm40"], dtype=object),
        base_hybrid=np.array(results["base_hybrid"], dtype=object)
    )

if __name__ == "__main__":
    generate_spaghetti_data()
