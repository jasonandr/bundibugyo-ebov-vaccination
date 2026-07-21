import sys
import os
import json
import time
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

scenarios = {
    "Exponential (shape=1.0, VE=0.45)": (1.0, 1.0, 0.45),
    "Gamma (shape=2.0, VE=0.45)": (2.0, 2.0, 0.45),
    "Gamma (shape=2.0, VE=0.55)": (2.0, 2.0, 0.55),
}

print("==========================================================================")
print("DIAGNOSTIC TEST: DRIVERS OF INCREMENTAL RING VAX BENEFIT")
print("==========================================================================")

for label, (inc_sh, inf_sh, ve) in scenarios.items():
    res_no_vax = []
    res_vax = []
    
    for rep in range(100):
        seed = 42 + rep
        # No Vax Enh Ops
        r_no = cpp_engine.run_simulation(
            rt_array_padded, 0.25, 8.5, 6.0, 1, 0.0, 10.0, 0.0,
            enh_rep, 1.0, 2.0, 1.0, -1, 100, 0, 0.454, 0.454,
            50, 50, len(rt_array), 0.0, False, 1.0, False,
            enh_trace, -1.0, -1.0, -1.0, 0.75, 2.0, False, True,
            0.0, 0, -1.0, 0.0, seed, 1.0, False, False, inc_sh, inf_sh, True
        )
        # Vax Enh Ops
        r_vax = cpp_engine.run_simulation(
            rt_array_padded, 0.25, 8.5, 6.0, 1, ve, 10.0, 0.8,
            enh_rep, 1.0, 2.0, 1.0, -1, 100, -1, 0.454, 0.454*(1.0-ve),
            50, 50, len(rt_array), 0.0, False, 1.0, False,
            enh_trace, -1.0, -1.0, -1.0, 0.75, 2.0, False, True,
            0.0, 0, -1.0, 0.0, seed, 1.0, False, False, inc_sh, inf_sh, True
        )
        res_no_vax.append(r_no[1] * N)
        res_vax.append(r_vax[1] * N)

    df_comp = pd.DataFrame({'no_vax': res_no_vax, 'vax': res_vax})
    inc_averted = (df_comp['no_vax'] - df_comp['vax']) / df_comp['no_vax'] * 100.0
    print(f"\nConfiguration: {label}")
    print(f"  No Vax Deaths: {np.median(df_comp['no_vax']):.0f} | Vax Deaths: {np.median(df_comp['vax']):.0f}")
    print(f"  Incremental Deaths Averted: {np.median(df_comp['no_vax'] - df_comp['vax']):.0f}")
    print(f"  Incremental % Deaths Averted: Median = {np.median(inc_averted):.1f}% (IQR: {np.percentile(inc_averted, 25):.1f}% - {np.percentile(inc_averted, 75):.1f}%)")
