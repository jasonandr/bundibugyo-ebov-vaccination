import os
import sys
import pandas as pd
from multiprocessing import Pool
import time

os.environ["FINAL_ESTIMATE_N"] = "100000"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from run_final_high_replicate_estimates import run_one, BASE_SEED

scenario = {
    "scenario": "vaccine_efficacy",
    "level": 0.2,
    "radius": 1,
    "efficacy": 0.2,
    "detection": 0.7,
    "uptake": 0.8,
    "max_vaccines": 1000000,
    "immune_onset_days": 10.0,
    "continuous_immune_onset": False,
    "sigmoidal_d0": None
}

if __name__ == "__main__":
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    
    # Calculate slice
    total_replicates = 5000
    num_jobs = 5
    replicates_per_job = total_replicates // num_jobs
    
    start_idx = task_id * replicates_per_job
    end_idx = start_idx + replicates_per_job
    
    print(f"Running missing scenario on array {task_id}: indices {start_idx} to {end_idx}")
    start = time.time()
    
    tasks = []
    base_seed = BASE_SEED + 999999 + start_idx
    for i in range(start_idx, end_idx):
        base_seed += 1
        tasks.append((scenario, i, base_seed))
        
    results = []
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=5)):
            results.append(res)
            if (i+1) % 100 == 0:
                print(f"Finished {i+1} / {replicates_per_job} tasks...", flush=True)
                
    df = pd.DataFrame(results)
    df.to_csv(f"data_and_results/final_high_replicate_raw_scg_missing_{task_id}.csv", index=False)
    print(f"Saved missing run chunk {task_id}! Took {time.time() - start:.1f}s")
