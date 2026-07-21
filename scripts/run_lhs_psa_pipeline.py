import os
import sys
import json
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd

from ebola_stochastic_ring import generate_network
import ebola_stochastic_ring_cpp as cpp

G_OFFSETS = None
G_EDGES = None
G_N = 100000

def init_worker(offsets, edges, N):
    global G_OFFSETS, G_EDGES, G_N
    G_OFFSETS = offsets
    G_EDGES = edges
    G_N = N

def process_sample_batch(item, fitted_params, rt_posterior_samples):
    idx, row = item
    sample_id = int(row['psa_sample_id'])
    ve = float(row['vaccine_efficacy'])
    
    rt_idx = int(row['rt_posterior_idx'])
    if rt_posterior_samples is not None and rt_idx < len(rt_posterior_samples):
        rt_s = rt_posterior_samples[rt_idx]
        rt_arr = list(rt_s) + [rt_s[-1]] * 40
    else:
        rt_base = fitted_params.get("Rt_array", [2.58]*50)
        rt_arr = list(rt_base) + [rt_base[-1]] * 40

    inc_per = float(row['incubation_period'])
    inf_per = float(row['infectious_period'])
    det_base = float(row['detection_delay_base'])
    det_enh = float(row['detection_delay_enh'])
    trace_enh = float(row['tracing_coverage_enh'])
    inc_shape = float(row['incubation_shape'])
    inf_shape = float(row['infectious_shape'])

    def make_ramp(target, duration=15, max_time=120):
        return np.linspace(0.3, target, duration).tolist() + [target]*(max_time-duration)

    base_rep = [0.3] * 120
    base_trace = [0.3] * 120
    enh_rep = make_ramp(0.7)
    enh_trace = make_ramp(trace_enh)

    # Base case ring vaccination uses Ring 2 (radius=2: first- and second-degree contacts)
    scenarios = {
        # 1. Base Operations Baseline
        "no_vax_base_ops": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'ring_radius': 2, 'efficacy': 0.0, 'uptake': 0.0, 'max_vaccines': 0
        },
        # 2. Ring Vax (Ring 2) under Base Ops
        "vax_base_ops": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'ring_radius': 2, 'efficacy': ve, 'uptake': 0.8
        },
        # 3. Enhanced Ops Alone
        "no_vax_enh_ops": {
            'reporting_rate': enh_rep, 'tracing_coverage': enh_trace,
            'detection_delay': det_enh, 'tracing_delay': 1.0,
            'ring_radius': 2, 'efficacy': 0.0, 'uptake': 0.0, 'max_vaccines': 0
        },
        # 4. Enhanced Ops + Ring Vax (Ring 2)
        "vax_enh_ops": {
            'reporting_rate': enh_rep, 'tracing_coverage': enh_trace,
            'detection_delay': det_enh, 'tracing_delay': 1.0,
            'ring_radius': 2, 'efficacy': ve, 'uptake': 0.8
        },
        # 5. Standalone Community Vaccination (under Base Ops)
        "comm_base_20": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'community_vax_coverage': 0.2, 'community_vax_trigger': 1, 'community_vax_delay': 0.0, 'community_vax_rollout_days': 14.0,
            'efficacy': ve, 'uptake': 0.8, 'max_vaccines': 0
        },
        "comm_base_40": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'community_vax_coverage': 0.4, 'community_vax_trigger': 1, 'community_vax_delay': 0.0, 'community_vax_rollout_days': 14.0,
            'efficacy': ve, 'uptake': 0.8, 'max_vaccines': 0
        },
        "comm_base_60": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'community_vax_coverage': 0.6, 'community_vax_trigger': 1, 'community_vax_delay': 0.0, 'community_vax_rollout_days': 14.0,
            'efficacy': ve, 'uptake': 0.8, 'max_vaccines': 0
        },
        "comm_base_80": {
            'reporting_rate': base_rep, 'tracing_coverage': base_trace,
            'detection_delay': det_base, 'tracing_delay': 2.0,
            'community_vax_coverage': 0.8, 'community_vax_trigger': 1, 'community_vax_delay': 0.0, 'community_vax_rollout_days': 14.0,
            'efficacy': ve, 'uptake': 0.8, 'max_vaccines': 0
        },
        # 6. Incremental Community Vaccination (under Enhanced Ops)
        "comm_enh_20": {
            'reporting_rate': enh_rep, 'tracing_coverage': enh_trace,
            'detection_delay': det_enh, 'tracing_delay': 1.0,
            'community_vax_coverage': 0.2, 'community_vax_trigger': 1, 'community_vax_delay': 0.0, 'community_vax_rollout_days': 14.0,
            'efficacy': ve, 'uptake': 0.8, 'max_vaccines': 0
        }
    }

    cpp_engine = cpp.EbolaEngine(G_N, G_OFFSETS, G_EDGES)
    n_reps_per_sample = 50
    max_sim_time = 90

    batch_results = []
    for rep in range(n_reps_per_sample):
        seed = 42 + rep
        for s_name, s_kwargs in scenarios.items():
            res = cpp_engine.run_simulation(
                rt_arr, 0.25, inc_per, inf_per,
                s_kwargs.get('ring_radius', 2), s_kwargs.get('efficacy', 0.0), 10.0, s_kwargs.get('uptake', 0.0),
                s_kwargs.get('reporting_rate', []), 1.0, s_kwargs.get('detection_delay', 4.0), s_kwargs.get('tracing_delay', 2.0),
                -1, 100, s_kwargs.get('max_vaccines', -1), 0.454, 0.454*(1.0-s_kwargs.get('efficacy', 0.0)),
                50, 50, max_sim_time, 0.0, False, 1.0, False,
                s_kwargs.get('tracing_coverage', []), -1.0, -1.0, -1.0,
                0.75, 2.0, False, True, s_kwargs.get('community_vax_coverage', 0.0), s_kwargs.get('community_vax_trigger', 0),
                s_kwargs.get('community_vax_delay', -1.0), s_kwargs.get('community_vax_rollout_days', 0.0),
                seed, 1.0, False, False, inc_shape, inf_shape, True
            )
            batch_results.append({
                'psa_sample_id': sample_id,
                'seed': seed,
                'scenario': s_name,
                'cases_count': res[0] * G_N,
                'deaths_count': res[1] * G_N,
                'vaccines_count': res[2],
                'vaccine_efficacy': ve,
                'incubation_period': inc_per,
                'infectious_period': inf_per
            })

    return batch_results

def run_psa_batch():
    print("Loading LHS parameter samples...")
    param_path = "../data_and_results/lhs_psa_parameter_samples.csv"
    if not os.path.exists(param_path):
        from lhs_parameter_sampler import generate_lhs_parameter_samples
        df_lhs = generate_lhs_parameter_samples(250)
        df_lhs.to_csv(param_path, index=False)
    else:
        df_lhs = pd.read_csv(param_path)

    with open("../data_and_results/fitted_parameters.json", "r") as f:
        fitted_params = json.load(f)

    rt_samples_file = "../data_and_results/rt_posterior_samples.npy"
    rt_posterior_samples = np.load(rt_samples_file) if os.path.exists(rt_samples_file) else None

    N = 100000
    n_samples = len(df_lhs)
    n_reps_per_sample = 50

    print(f"==========================================================")
    print(f"PARALLEL RING 2 PSA SIMULATION PIPELINE (N={N})")
    print(f"LHS Draws: {n_samples}, Replicates per Draw: {n_reps_per_sample}")
    print(f"Total Simulations per Scenario: {n_samples * n_reps_per_sample:,}")
    print(f"Ring Radius: Ring 2 (Radius 2)")
    print(f"==========================================================")

    print("Generating baseline 100,000-node contact network...")
    G = generate_network(N, household_mean=5.2, community_mean=30.0, community_variance=160.0)

    offsets = np.zeros(N + 1, dtype=np.int32)
    edges = []
    for i in range(N):
        offsets[i] = len(edges)
        edges.extend([int(x) for x in G.neighbors(i)])
    offsets[N] = len(edges)
    edges = np.array(edges, dtype=np.int32)

    t_start = time.time()
    n_cpus = mp.cpu_count()
    print(f"Launching parallel execution pool with {n_cpus} workers...")

    items = list(df_lhs.iterrows())
    worker_fn = partial(process_sample_batch, fitted_params=fitted_params, rt_posterior_samples=rt_posterior_samples)

    with mp.Pool(processes=n_cpus, initializer=init_worker, initargs=(offsets, edges, N)) as pool:
        results_nested = pool.map(worker_fn, items)

    all_results = [item for sublist in results_nested for item in sublist]
    elapsed = time.time() - t_start
    print(f"Parallel simulation pipeline completed in {elapsed:.2f} seconds!")

    df_results = pd.DataFrame(all_results)
    out_raw = "../data_and_results/psa_raw_simulations.csv"
    df_results.to_csv(out_raw, index=False)

    print("Calculating Expected Value PSA Summaries...")
    df_ev = df_results.groupby(['psa_sample_id', 'scenario'])[['cases_count', 'deaths_count', 'vaccines_count']].mean().reset_index()

    base_ev = df_ev[df_ev['scenario'] == 'no_vax_base_ops'].set_index('psa_sample_id')
    enh_ev = df_ev[df_ev['scenario'] == 'no_vax_enh_ops'].set_index('psa_sample_id')
    
    psa_summary_rows = []
    
    for s_name in df_ev['scenario'].unique():
        s_ev = df_ev[df_ev['scenario'] == s_name].set_index('psa_sample_id')
        merged = s_ev.join(base_ev, lsuffix='_interv', rsuffix='_base')
        
        deaths_averted_pct = (merged['deaths_count_base'] - merged['deaths_count_interv']) / merged['deaths_count_base'] * 100.0
        abs_deaths_averted = merged['deaths_count_base'] - merged['deaths_count_interv']

        psa_summary_rows.append({
            'scenario': s_name,
            'comparator': 'vs_base_ops',
            'median_deaths_count': np.median(merged['deaths_count_interv']),
            'median_deaths_averted_abs': np.median(abs_deaths_averted),
            'median_deaths_averted_pct': np.median(deaths_averted_pct),
            'psa_ui_low_95': np.percentile(deaths_averted_pct, 2.5),
            'psa_ui_high_95': np.percentile(deaths_averted_pct, 97.5),
            'iqr_low_25': np.percentile(deaths_averted_pct, 25.0),
            'iqr_high_75': np.percentile(deaths_averted_pct, 75.0)
        })

    # Add Incremental Ring Vax (vax_enh_ops vs no_vax_enh_ops)
    ring_ev = df_ev[df_ev['scenario'] == 'vax_enh_ops'].set_index('psa_sample_id')
    merged_ring = ring_ev.join(enh_ev, lsuffix='_ring', rsuffix='_enh')
    ring_averted_pct = (merged_ring['deaths_count_enh'] - merged_ring['deaths_count_ring']) / merged_ring['deaths_count_enh'] * 100.0
    ring_abs_averted = merged_ring['deaths_count_enh'] - merged_ring['deaths_count_ring']

    psa_summary_rows.append({
        'scenario': 'incremental_ring_vax',
        'comparator': 'vs_enh_ops',
        'median_deaths_count': np.median(merged_ring['deaths_count_ring']),
        'median_deaths_averted_abs': np.median(ring_abs_averted),
        'median_deaths_averted_pct': np.median(ring_averted_pct),
        'psa_ui_low_95': np.percentile(ring_averted_pct, 2.5),
        'psa_ui_high_95': np.percentile(ring_averted_pct, 97.5),
        'iqr_low_25': np.percentile(ring_averted_pct, 25.0),
        'iqr_high_75': np.percentile(ring_averted_pct, 75.0)
    })

    # Add Incremental Community Vax 20% (comm_enh_20 vs no_vax_enh_ops)
    comm_enh_ev = df_ev[df_ev['scenario'] == 'comm_enh_20'].set_index('psa_sample_id')
    merged_comm_inc = comm_enh_ev.join(enh_ev, lsuffix='_comm', rsuffix='_enh')
    comm_inc_averted_pct = (merged_comm_inc['deaths_count_enh'] - merged_comm_inc['deaths_count_comm']) / merged_comm_inc['deaths_count_enh'] * 100.0
    comm_inc_abs_averted = merged_comm_inc['deaths_count_enh'] - merged_comm_inc['deaths_count_comm']

    psa_summary_rows.append({
        'scenario': 'incremental_comm_vax_20',
        'comparator': 'vs_enh_ops',
        'median_deaths_count': np.median(merged_comm_inc['deaths_count_comm']),
        'median_deaths_averted_abs': np.median(comm_inc_abs_averted),
        'median_deaths_averted_pct': np.median(comm_inc_averted_pct),
        'psa_ui_low_95': np.percentile(comm_inc_averted_pct, 2.5),
        'psa_ui_high_95': np.percentile(comm_inc_averted_pct, 97.5),
        'iqr_low_25': np.percentile(comm_inc_averted_pct, 25.0),
        'iqr_high_75': np.percentile(comm_inc_averted_pct, 75.0)
    })

    df_summary = pd.DataFrame(psa_summary_rows)
    out_summary = "../data_and_results/psa_summary_results.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"Saved Ring 2 PSA Summary to {out_summary}")

    print("\n==========================================================================")
    print("RING 2 PSA 95% PARAMETER UNCERTAINTY INTERVAL RESULTS")
    print("==========================================================================")
    for idx, r in df_summary.iterrows():
        print(f"Scenario: {r['scenario']:<25} ({r['comparator']:<12}) | % Averted: {r['median_deaths_averted_pct']:5.1f}% | 95% UI: [{r['psa_ui_low_95']:5.1f}% - {r['psa_ui_high_95']:5.1f}%]")

if __name__ == '__main__':
    run_psa_batch()
