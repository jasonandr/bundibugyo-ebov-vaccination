import os
import sys

# VERY IMPORTANT: Set N to 100k before importing
os.environ["FINAL_ESTIMATE_N"] = "100000"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import multiprocessing
import time
from run_final_high_replicate_estimates import run_one, BASE_SEED

target_scenarios = []
for cap in [10000, 20000, 60000]:
    for r in [1, 2]:
        target_scenarios.append({
            "scenario": "stockpile_cap",
            "level": str(cap),
            "radius": r,
            "efficacy": 0.4,
            "detection": 0.7,
            "uptake": 0.8,
            "max_vaccines": cap,
            "immune_onset_days": 10.0,
            "continuous_immune_onset": False,
            "sigmoidal_d0": None
        })

print(f"Running {len(target_scenarios)} scenarios locally for N=100000...")

if __name__ == "__main__":
    start = time.time()
    tasks = []
    base_seed = BASE_SEED
    for sc in target_scenarios:
        for i in range(5000):
            base_seed += 1
            tasks.append((sc, i, base_seed))
            
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    with multiprocessing.Pool(12) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=10)):
            results.append(res)
            if (i+1) % 1000 == 0:
                print(f"Finished {i+1} / {len(tasks)} tasks...")
                
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(__file__), "../data_and_results/final_high_replicate_raw_scg_array_stockpile_new.csv"), index=False)
    print(f"Saved local stockpile runs! Took {time.time() - start:.1f}s")
