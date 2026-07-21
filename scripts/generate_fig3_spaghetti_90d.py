import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, repo_root)

from ebola_stochastic_ring import generate_network
import ebola_stochastic_ring_cpp as cpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-id", type=int, default=1)
    args = parser.parse_args()

    N = 100000
    G = generate_network(N, household_mean=5.2, community_mean=30.0, community_variance=160.0)

    offsets = np.zeros(N + 1, dtype=np.int32)
    edges = []
    for i in range(N):
        offsets[i] = len(edges)
        edges.extend([int(x) for x in G.neighbors(i)])
    offsets[N] = len(edges)
    edges = np.array(edges, dtype=np.int32)

    cpp_engine = cpp.EbolaEngine(N, offsets, edges)

    param_file = os.path.join(repo_root, "data_and_results", "fitted_parameters.json")
    with open(param_file, "r") as f:
        params = json.load(f)
    rt_array = params.get("Rt_array", [2.58]*50)
    rt_arr_padded = list(rt_array) + [rt_array[-1]] * 40

    def make_ramp(target, duration=15, max_time=120):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)

    base_rep = [0.3] * 120
    base_trace = [0.3] * 120
    enh_rep = make_ramp(0.7)
    enh_trace = make_ramp(0.8)

    scenarios = {
        "no_vax_base_ops": {'rep': base_rep, 'trace': base_trace, 'det': 4.0, 'ring': 2, 've': 0.0, 'comm': 0.0},
        "vax_base_ops": {'rep': base_rep, 'trace': base_trace, 'det': 4.0, 'ring': 2, 've': 0.45, 'comm': 0.0},
        "no_vax_enh_ops": {'rep': enh_rep, 'trace': enh_trace, 'det': 2.0, 'ring': 2, 've': 0.0, 'comm': 0.0},
        "vax_enh_ops": {'rep': enh_rep, 'trace': enh_trace, 'det': 2.0, 'ring': 2, 've': 0.45, 'comm': 0.0},
        "comm_base_20": {'rep': base_rep, 'trace': base_trace, 'det': 4.0, 'ring': 2, 've': 0.45, 'comm': 0.2},
    }

    n_reps = 10
    sim_data = []

    for s_name, s_cfg in scenarios.items():
        for r in range(n_reps):
            seed = args.array_id * 1000 + r
            res = cpp_engine.run_simulation(
                rt_arr_padded, 0.25, 8.5, 6.0,
                s_cfg['ring'], s_cfg['ve'], 10.0, 0.8,
                s_cfg['rep'], 1.0, s_cfg['det'], 2.0,
                -1, 100, -1, 0.454, 0.454*(1.0-s_cfg['ve']),
                50, 50, 90, 0.0, False, 1.0, False,
                s_cfg['trace'], -1.0, -1.0, -1.0,
                0.75, 2.0, False, True, s_cfg['comm'], 1 if s_cfg['comm']>0 else 0,
                0.0, 14.0, seed, 1.0, False, False, 1.8, 2.0, True
            )
            sim_data.append({
                'array_id': args.array_id,
                'rep': r,
                'scenario': s_name,
                'cases': res[0] * N,
                'deaths': res[1] * N,
                'vaccines': res[2]
            })

    out_dir = os.path.join(repo_root, "data_and_results", "fig3_chunks")
    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(sim_data)
    df_out.to_csv(os.path.join(out_dir, f"fig3_chunk_{args.array_id}.csv"), index=False)
    print(f"Finished Fig 3 task {args.array_id}")

if __name__ == '__main__':
    main()
