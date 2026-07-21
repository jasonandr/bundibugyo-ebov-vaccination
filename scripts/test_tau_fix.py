import json
import numpy as np

with open("data_and_results/fitted_parameters.json") as f:
    RT_ARRAY = json.load(f)["Rt_array"]

with open("data_and_results/rt_calibrated_tau_array.json") as f:
    TAU_ARRAY = json.load(f)["tau_array"]
    
print("Max Rt:", max(RT_ARRAY))
print("Max Tau:", max(TAU_ARRAY))
