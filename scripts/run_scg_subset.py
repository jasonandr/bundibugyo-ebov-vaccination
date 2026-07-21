import os
import sys
import pandas as pd
from multiprocessing import Pool
import time

os.environ["FINAL_ESTIMATE_N"] = "100000"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from run_final_high_replicate_estimates import run_one, BASE_SEED, scenario_definitions

import argparse

def get_subset_scenarios(subset_name):
    scenarios = scenario_definitions()
    
    if subset_name == "missing_baselines":
        # Indices 0 to 4 contain the 4 no_vaccination scenarios and the VE=0.2, Radius=1 scenario
        return scenarios[0:5]
    elif subset_name == "scaled_cfr":
        return [s for s in scenarios if s["scenario"] == "vaccine_efficacy_scaled_cfr"]
    else:
        raise ValueError(f"Unknown subset: {subset_name}")

if __name__ == "__main__":
    subset_name = os.environ.get("SCG_SUBSET_NAME", "missing_baselines")
    scenarios = get_subset_scenarios(subset_name)
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    
    total_replicates = 5000
    num_jobs = 25
    replicates_per_job = total_replicates // num_jobs
    
    start_idx = task_id * replicates_per_job
    end_idx = start_idx + replicates_per_job
    
    print(f"Running subset {subset_name} array {task_id}: indices {start_idx} to {end_idx}")
    start = time.time()
    
    tasks = []
    base_seed = BASE_SEED + start_idx
    
    for i in range(start_idx, end_idx):
        base_seed += 1
        for sc in scenarios:
            tasks.append((sc, i, base_seed))
            
    results = []
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=10)):
            results.append(res)
            if (i+1) % 100 == 0:
                print(f"Finished {i+1} / {len(tasks)} tasks...", flush=True)
                
    df = pd.DataFrame(results)
    df.to_csv(f"data_and_results/final_high_replicate_raw_scg_{subset_name}_{task_id}.csv", index=False)
    print(f"Saved {subset_name} chunk {task_id}! Took {time.time() - start:.1f}s")
