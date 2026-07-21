import numpy as np
import pandas as pd
import json
import os
from scipy.stats import qmc, norm, uniform

def generate_lhs_parameter_samples(n_samples=250, seed=20260721):
    """
    Generates n_samples parameter sets using Latin Hypercube Sampling (LHS)
    across the joint probability distributions of model parameters.
    """
    # 10 parameters to sample
    d = 10
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample_unit = sampler.random(n=n_samples)
    
    # 1. Vaccine Efficacy (VE): Uniform [0.30, 0.60] centered at 0.45
    ve_samples = qmc.scale(sample_unit[:, 0:1], [0.30], [0.60]).flatten()
    
    # 2. Rt trajectory index: Integer [0, 999] drawing from EpiNow2 posterior
    rt_idx_samples = np.floor(sample_unit[:, 1] * 1000).astype(int)
    rt_idx_samples = np.clip(rt_idx_samples, 0, 999)
    
    # 3. Mean Incubation Period: Normal(8.5, 1.0) clipped [6.0, 11.0]
    inc_mean_samples = norm.ppf(0.05 + 0.90 * sample_unit[:, 2], loc=8.5, scale=1.0)
    
    # 4. Mean Infectious Period: Normal(6.0, 0.8) clipped [4.0, 8.0]
    inf_mean_samples = norm.ppf(0.05 + 0.90 * sample_unit[:, 3], loc=6.0, scale=0.8)
    
    # 5. Detection Delay Base: Uniform [3.0, 5.0]
    det_base_samples = qmc.scale(sample_unit[:, 4:5], [3.0], [5.0]).flatten()
    
    # 6. Detection Delay Enh: Uniform [1.5, 3.5]
    det_enh_samples = qmc.scale(sample_unit[:, 5:6], [1.5], [3.5]).flatten()
    
    # 7. Tracing Coverage Enh: Uniform [0.60, 0.90]
    trace_enh_samples = qmc.scale(sample_unit[:, 6:7], [0.60], [0.90]).flatten()
    
    # 8. Community Variance (Overdispersion): Uniform [80.0, 240.0] centered at 160.0
    var_samples = qmc.scale(sample_unit[:, 7:8], [80.0], [240.0]).flatten()
    
    # 9. Incubation Shape (Gamma vs Exp): Uniform [1.0, 3.0]
    inc_shape_samples = qmc.scale(sample_unit[:, 8:9], [1.0], [3.0]).flatten()
    
    # 10. Infectious Shape (Gamma vs Exp): Uniform [1.0, 3.0]
    inf_shape_samples = qmc.scale(sample_unit[:, 9:10], [1.0], [3.0]).flatten()
    
    df_lhs = pd.DataFrame({
        'psa_sample_id': np.arange(n_samples),
        'vaccine_efficacy': ve_samples,
        'rt_posterior_idx': rt_idx_samples,
        'incubation_period': inc_mean_samples,
        'infectious_period': inf_mean_samples,
        'detection_delay_base': det_base_samples,
        'detection_delay_enh': det_enh_samples,
        'tracing_coverage_enh': trace_enh_samples,
        'community_variance': var_samples,
        'incubation_shape': inc_shape_samples,
        'infectious_shape': inf_shape_samples
    })
    
    return df_lhs

if __name__ == '__main__':
    df = generate_lhs_parameter_samples(250)
    os.makedirs('../data_and_results', exist_ok=True)
    out_path = '../data_and_results/lhs_psa_parameter_samples.csv'
    df.to_csv(out_path, index=False)
    print(f"Generated 250 Latin Hypercube Parameter Samples -> {out_path}")
    print(df.head())
