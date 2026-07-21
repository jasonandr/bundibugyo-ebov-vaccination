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
    print(f"Running missing scenario: {scenario}")
    start = time.time()
    tasks = []
    base_seed = BASE_SEED + 999999
    for i in range(5000):
        base_seed += 1
        tasks.append((scenario, i, base_seed))
        
    results = []
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=10)):
            results.append(res)
            if (i+1) % 500 == 0:
                print(f"Finished {i+1} / 5000 tasks...", flush=True)
                
    df = pd.DataFrame(results)
    df.to_csv("data_and_results/final_high_replicate_raw_scg_missing.csv", index=False)
    print(f"Saved missing run! Took {time.time() - start:.1f}s")
