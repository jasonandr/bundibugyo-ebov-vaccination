import numpy as np
from scipy.stats import gamma
import pandas as pd

def fast_rt_regularized(cases_inc, window=7):
    N_sims, T = cases_inc.shape
    
    # 7-day smoothing
    kernel = np.ones(window) / window
    cases_inc_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full')[:T], axis=1, arr=cases_inc)
    
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, T+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    alpha_prior = 2.25
    beta_prior = 1.5
    
    Rt = np.zeros((N_sims, T))
    for t in range(1, T):
        Lambda = np.zeros(N_sims)
        for tau in range(1, t+1):
            if tau <= len(w):
                Lambda += cases_inc_smooth[:, t-tau] * w[tau-1]
                
        Rt[:, t] = (alpha_prior + cases_inc_smooth[:, t]) / (beta_prior + Lambda)
        
    return Rt

z = np.load("../data_and_results/new_spaghetti_chunks/chunk_1.npz", allow_pickle=True)
base_no_vax = z["base_no_vax"][:100]
rt = fast_rt_regularized(base_no_vax[:, :90].astype(float))
print("Max Rt:", np.max(rt))
print("Min Rt:", np.min(rt))
