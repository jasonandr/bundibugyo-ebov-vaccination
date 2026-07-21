import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    with open("../data_and_results/fitted_parameters.json", "r") as f:
        params = json.load(f)
    rt_array = params.get("Rt_array", [2.58]*50)
    rt_arr_padded = list(rt_array) + [rt_array[-1]] * 40

    def make_ramp(target, duration=15, max_time=120):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)

    enh_rep = make_ramp(0.7)
    enh_trace = make_ramp(0.8)

    delays = [0.0, 7.0, 14.0, 21.0]
    coverages = [0.1, 0.2, 0.4, 0.6, 0.8]

    chunk_data = []
    for delay in delays:
        for cov in coverages:
            for r in range(5):
                seed = args.array_id * 500 + r
                res = cpp_engine.run_simulation(
                    rt_arr_padded, 0.25, 8.5, 6.0,
                    2, 0.45, 10.0, 0.8,
                    enh_rep, 1.0, 2.0, 1.0,
                    -1, 100, -1, 0.454, 0.454*0.55,
                    50, 50, 90, 0.0, False, 1.0, False,
                    enh_trace, -1.0, -1.0, -1.0,
                    0.75, 2.0, False, True, cov, 1,
                    delay, 14.0, seed, 1.0, False, False, 1.8, 2.0, True
                )
                chunk_data.append({
                    'array_id': args.array_id,
                    'delay': delay, 'coverage': cov,
                    'deaths': res[1] * N,
                    'vaccines': res[2]
                })

    out_dir = "../data_and_results/fig5_chunks"
    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(chunk_data)
    df_out.to_csv(f"{out_dir}/fig5_chunk_{args.array_id}.csv", index=False)
    print(f"Finished Fig 5 task {args.array_id}")

if __name__ == '__main__':
    main()
