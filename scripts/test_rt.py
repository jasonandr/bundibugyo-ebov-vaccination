import numpy as np
from scipy.stats import gamma

def fast_rt(cases_inc, window=7):
    # cases_inc is (N_sims, T)
    N_sims, T = cases_inc.shape
    
    # Pre-smooth incidence with a rolling average (causal)
    # Using a simple convolution for rolling mean along time axis
    kernel = np.ones(window) / window
    # Pad left with 0s to keep it causal and same size
    cases_inc_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full')[:T], axis=1, arr=cases_inc)
    
    # Generation time distribution
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, T+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    Rt = np.zeros((N_sims, T))
    for t in range(1, T):
        # Lambda_t = sum(w[tau]*I[t-tau])
        # We can compute Lambda efficiently
        Lambda = np.zeros(N_sims)
        for tau in range(1, t+1):
            Lambda += cases_inc_smooth[:, t-tau] * w[tau-1]
        
        mask = Lambda > 0
        Rt[mask, t] = cases_inc_smooth[mask, t] / Lambda[mask]
        
    return Rt

# Test with dummy data
np.random.seed(42)
dummy_inc = np.random.poisson(lam=10, size=(100, 90))
rt = fast_rt(dummy_inc)
print("Shape:", rt.shape)
print("Mean Rt at day 50:", np.mean(rt[:, 50]))
