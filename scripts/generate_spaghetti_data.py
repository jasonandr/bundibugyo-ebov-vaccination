import os
import sys
import numpy as np
import ebola_stochastic_ring as sim
from paths import result_path
import json

def generate_spaghetti_data():
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    reps_per_job = 1000
    _G = sim.generate_network(10000)
    
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = json.load(f)["tau_array"]
    max_tau = max(TAU_ARRAY)
    RT_ARRAY = [t * (6.066 / max_tau) for t in TAU_ARRAY]
    
    def make_ramp(target, duration=15, max_time=91):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)
        
    mod_ops = make_ramp(0.5)
    opt_rep = make_ramp(0.7)
    opt_tr = make_ramp(0.8)
    
    results = {
        "no_vax_mod": [],
        "no_vax_opt": [],
        "ring2_mod": [],
        "ring2_opt": [],
        "comm_40": [],
        "hybrid": []
    }
    
    for i in range(reps_per_job):
        seed = array_id * 100000 + i
        
        # Baselines
        nv_mod = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.0, reporting_rate=mod_ops, tracing_coverage=mod_ops,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        nv_opt = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.0, reporting_rate=opt_rep, tracing_coverage=opt_tr,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        # 1. Radius 2 ring (moderate)
        r2_mod = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.9, reporting_rate=mod_ops, tracing_coverage=mod_ops,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        # 2. Radius 2 ring (optimistic)
        r2_opt = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.9, reporting_rate=opt_rep, tracing_coverage=opt_tr,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        # 3. Community Vax (40%) - optimistic ops is standard for community mass vax base
        c40 = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, max_vaccines=0,
            vaccine_effect=0.9, reporting_rate=opt_rep, tracing_coverage=opt_tr,
            community_vax_coverage=0.4, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        # 4. Hybrid (Community + radius 2 with optimistic ops)
        hyb = sim.simulate_ring_vaccination(
            _G, initial_infected=5, rt_array=RT_ARRAY, baseline_tau=0.25, ring_radius=2, 
            vaccine_effect=0.9, reporting_rate=opt_rep, tracing_coverage=opt_tr,
            community_vax_coverage=0.4, community_vax_trigger=1, community_vax_delay=0.0,
            community_vax_rollout_days=14.0,
            max_sim_time=90, seed=seed, return_time_series=True, engine='cpp'
        )
        
        results["no_vax_mod"].append(np.array(nv_mod)) # just track deaths over time
        results["no_vax_opt"].append(np.array(nv_opt))
        results["ring2_mod"].append(np.array(r2_mod))
        results["ring2_opt"].append(np.array(r2_opt))
        results["comm_40"].append(np.array(c40))
        results["hybrid"].append(np.array(hyb))

    out_dir = "data_and_results/spaghetti_chunks"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        f"{out_dir}/chunk_{array_id}.npz",
        no_vax_mod=np.array(results["no_vax_mod"]),
        no_vax_opt=np.array(results["no_vax_opt"]),
        ring2_mod=np.array(results["ring2_mod"]),
        ring2_opt=np.array(results["ring2_opt"]),
        comm_40=np.array(results["comm_40"]),
        hybrid=np.array(results["hybrid"])
    )

if __name__ == "__main__":
    generate_spaghetti_data()
