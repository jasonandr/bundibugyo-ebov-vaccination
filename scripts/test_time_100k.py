import os
import sys
os.environ["FINAL_ESTIMATE_N"] = "100000"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
from run_final_high_replicate_estimates import run_one, BASE_SEED

sc = {
    "scenario": "stockpile_cap",
    "level": "10000",
    "radius": 1,
    "efficacy": 0.4,
    "detection": 0.7,
    "uptake": 0.8,
    "max_vaccines": 10000,
    "immune_onset_days": 10.0,
    "continuous_immune_onset": False,
    "sigmoidal_d0": None
}

start = time.time()
print("Starting 1 replicate of stockpile_cap 10000...", flush=True)
res = run_one((sc, 0, BASE_SEED))
end = time.time()
print(f"Finished in {end - start:.2f} seconds.", flush=True)
