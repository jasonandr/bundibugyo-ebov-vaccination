import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/jasonandrews/repos/ebola vaccination modeling/scripts')

from ebola_stochastic_ring import generate_network
import ebola_stochastic_ring_cpp as cpp

# Load fitted Rt array
with open('/Users/jasonandrews/repos/ebola vaccination modeling/data_and_results/fitted_parameters.json', 'r') as f:
    params = json.load(f)
rt_array = params.get('Rt_array', [2.58]*50)
rt_array_padded = list(rt_array) + [rt_array[-1]] * 30

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

def make_ramp(target, duration=15, max_time=91):
    return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)

enh_rep = make_ramp(0.7)
enh_trace = make_ramp(0.8)

# Test incubation shape 1.0 (Exponential) vs 3.0 (Gamma) under tracing delays of 1.0d and 3.0d
scenarios = {
    "Exp (shape=1.0, trace_delay=1.0d)": (1.0, 1.0),
    "Exp (shape=1.0, trace_delay=3.0d)": (1.0, 3.0),
    "Gamma (shape=3.0, trace_delay=1.0d)": (3.0, 1.0),
    "Gamma (shape=3.0, trace_delay=3.0d)": (3.0, 3.0),
}

print("==========================================================================")
print("MECHANISTIC PROOF: GAMMA LATENCY WINDOW & RING VAX EFFICACY")
print("==========================================================================")

for label, (sh, t_del) in scenarios.items():
    averted_list = []
    for rep in range(50):
        seed = 42 + rep
        # No vax
        r_no = cpp_engine.run_simulation(
            rt_array_padded, 0.25, 8.5, 6.0, 1, 0.0, 10.0, 0.0,
            enh_rep, 1.0, 2.0, t_del, -1, 100, 0, 0.454, 0.454,
            50, 50, len(rt_array), 0.0, False, 1.0, False,
            enh_trace, -1.0, -1.0, -1.0, 0.75, 2.0, False, True,
            0.0, 0, -1.0, 0.0, seed, 1.0, False, False, sh, sh, True
        )
        # Ring Vax
        r_vax = cpp_engine.run_simulation(
            rt_array_padded, 0.25, 8.5, 6.0, 1, 0.45, 10.0, 0.8,
            enh_rep, 1.0, 2.0, t_del, -1, 100, -1, 0.454, 0.454*0.55,
            50, 50, len(rt_array), 0.0, False, 1.0, False,
            enh_trace, -1.0, -1.0, -1.0, 0.75, 2.0, False, True,
            0.0, 0, -1.0, 0.0, seed, 1.0, False, False, sh, sh, True
        )
        deaths_no = r_no[1] * N
        deaths_vax = r_vax[1] * N
        averted_list.append((deaths_no - deaths_vax) / deaths_no * 100.0)

    med = np.median(averted_list)
    print(f"  {label:<38} -> Incremental Ring Vax Reduction: {med:5.1f}%")
