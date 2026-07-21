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

    det_grid = [0.3, 0.5, 0.7, 0.9]
    trace_grid = [0.3, 0.5, 0.7, 0.9]
    ve_grid = [0.3, 0.45, 0.6, 0.8]

    chunk_data = []
    idx = 0
    for det in det_grid:
        for trace in trace_grid:
            for ve in ve_grid:
                idx += 1
                if (idx % 50) != (args.array_id % 50):
                    continue
                rep_arr = make_ramp(det)
                trace_arr = make_ramp(trace)
                
                for r in range(5):
                    seed = args.array_id * 100 + r
                    res = cpp_engine.run_simulation(
                        rt_arr_padded, 0.25, 8.5, 6.0,
                        2, ve, 10.0, 0.8,
                        rep_arr, 1.0, 2.0, 1.0,
                        -1, 100, -1, 0.454, 0.454*(1.0-ve),
                        50, 50, 90, 0.0, False, 1.0, False,
                        trace_arr, -1.0, -1.0, -1.0,
                        0.75, 2.0, False, True, 0.0, 0,
                        0.0, 14.0, seed, 1.0, False, False, 1.8, 2.0, True
                    )
                    chunk_data.append({
                        'array_id': args.array_id,
                        'det': det, 'trace': trace, 've': ve,
                        'deaths': res[1] * N
                    })

    out_dir = os.path.join(repo_root, "data_and_results", "fig4_chunks")
    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(chunk_data)
    df_out.to_csv(os.path.join(out_dir, f"fig4_chunk_{args.array_id}.csv"), index=False)
    print(f"Finished Fig 4 task {args.array_id}")

if __name__ == '__main__':
    main()
