import os
import sys
import pandas as pd
from multiprocessing import Pool
import time

os.environ["FINAL_ESTIMATE_N"] = "100000"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from run_final_high_replicate_estimates import run_one, BASE_SEED

# 5 stockpile levels
levels = [500, 3000, 10000, 20000, 60000]

scenarios = []
for level in levels:
    # 1. Radius 1
    scenarios.append({
        "scenario": "stockpile_cap_radius1",
        "level": level,
        "radius": 1,
        "efficacy": 0.6,
        "detection": 0.7,
        "uptake": 0.8,
        "max_vaccines": level,
        "immune_onset_days": 10.0,
        "continuous_immune_onset": False,
        "sigmoidal_d0": None,
        "compete_queue": False
    })
    
    # 2. Radius 2 Protected
    scenarios.append({
        "scenario": "stockpile_cap_radius2_protected",
        "level": level,
        "radius": 2,
        "efficacy": 0.6,
        "detection": 0.7,
        "uptake": 0.8,
        "max_vaccines": level,
        "immune_onset_days": 10.0,
        "continuous_immune_onset": False,
        "sigmoidal_d0": None,
        "compete_queue": False
    })
    
    # 3. Radius 2 Competing
    scenarios.append({
        "scenario": "stockpile_cap_radius2_competing",
        "level": level,
        "radius": 2,
        "efficacy": 0.6,
        "detection": 0.7,
        "uptake": 0.8,
        "max_vaccines": level,
        "immune_onset_days": 10.0,
        "continuous_immune_onset": False,
        "sigmoidal_d0": None,
        "compete_queue": True
    })

if __name__ == "__main__":
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    
    total_replicates = 5000
    num_jobs = 25
    replicates_per_job = total_replicates // num_jobs
    
    start_idx = task_id * replicates_per_job
    end_idx = start_idx + replicates_per_job
    
    print(f"Running dilution array {task_id}: indices {start_idx} to {end_idx}")
    start = time.time()
    
    tasks = []
    base_seed = BASE_SEED + 777777 + start_idx
    
    for i in range(start_idx, end_idx):
        base_seed += 1
        for sc in scenarios:
            tasks.append((sc, i, base_seed))
            
    results = []
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=10)):
            results.append(res)
            if (i+1) % 500 == 0:
                print(f"Finished {i+1} / {len(tasks)} tasks...", flush=True)
                
    df = pd.DataFrame(results)
    df.to_csv(f"data_and_results/final_high_replicate_raw_scg_dilution_{task_id}.csv", index=False)
    print(f"Saved dilution run chunk {task_id}! Took {time.time() - start:.1f}s")
