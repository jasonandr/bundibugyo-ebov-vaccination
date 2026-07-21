import json
import numpy as np
from ebola_stochastic_ring import calibrate_tau, generate_network
from paths import result_path

def main():
    print("Loading parameters...")
    with open(result_path("fitted_parameters.json")) as f:
        params = json.load(f)
    
    rt_array = params["Rt_array"]
    
    print("Generating base network...")
    G = generate_network(100000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    
    print("Calibrating tau array (this takes a moment)...")
    tau_array = []
    for i, rt in enumerate(rt_array):
        print(f"Calibrating for Rt={rt:.2f} (Day {i}/{len(rt_array)})...")
        tau = calibrate_tau(G, target_R0=rt, gamma=1.0/6.0, num_trials=30)
        tau_array.append(tau)
        
    out = {"tau_array": tau_array}
    out_path = result_path("rt_calibrated_tau_array.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
        
    print(f"Saved calibrated tau array to {out_path}")

if __name__ == "__main__":
    main()
