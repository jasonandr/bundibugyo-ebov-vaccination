import os
import sys
from multiprocessing import Pool
import numpy as np

from run_final_high_replicate_estimates import (
    scenario_definitions, load_existing_rows, run_one, 
    write_csv, RAW_FIELDS, N_REPLICATES, POPULATION_SIZE,
    result_path, N_WORKERS, BASE_SEED
)

def main():
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print("Error: SLURM_ARRAY_TASK_ID is not set.")
        sys.exit(1)
        
    task_id = int(task_id_str)
    scenarios = scenario_definitions()
    
    if task_id < 0 or task_id >= len(scenarios):
        print(f"Error: task_id {task_id} is out of bounds (0-{len(scenarios)-1}).")
        sys.exit(1)
        
    scenario = scenarios[task_id]
    
    # Each array task writes to its own raw CSV file
    raw_path = result_path(f"final_high_replicate_raw_scg_array_{task_id}.csv")
    all_rows = load_existing_rows(raw_path)
    
    seed = max([BASE_SEED] + [row["seed"] for row in all_rows]) if all_rows else BASE_SEED
    
    label = f"{scenario['scenario']} level={scenario['level']} radius={scenario['radius']}"
    completed = [
        row for row in all_rows
        if row["scenario"] == scenario["scenario"]
        and row["level"] == scenario["level"]
        and row["radius"] == scenario["radius"]
    ]
    
    if len(completed) >= N_REPLICATES:
        print(f"Skipping {label}; {len(completed)} replicates already complete", flush=True)
        return

    print(
        f"Running array task {task_id}: {label} with {N_REPLICATES - len(completed)} remaining "
        f"of {N_REPLICATES} replicates at N={POPULATION_SIZE}",
        flush=True,
    )
    
    args = []
    for replicate in range(len(completed), N_REPLICATES):
        seed += 1
        args.append((scenario, replicate, seed))
        
    with Pool(processes=N_WORKERS) as pool:
        rows = []
        for idx, row in enumerate(pool.imap_unordered(run_one, args, chunksize=10), start=1):
            rows.append(row)
            if idx % 1000 == 0:
                print(f"  completed {idx}/{len(args)}", flush=True)
                
    all_rows.extend(rows)
    write_csv(raw_path, all_rows, RAW_FIELDS)

if __name__ == "__main__":
    main()
