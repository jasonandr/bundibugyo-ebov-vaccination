import numpy as np
import EoN
from scipy.stats import nbinom

def get_final_size_ebcm(R0, infectious_period, coverage, efficacy, N=100000, initial_infected=10):
    """
    Simulates the Ebola outbreak using Edge-Based Compartmental Modeling.
    Returns the final fraction of the population infected.
    
    For tipping points and final epidemic size, SIR yields the same final size as SEIR
    for a given network and R0, so we use EoN's optimized SIR EBCM.
    
    Leaky vaccination is approximated by scaling the average transmission probability 
    per contact across the network.
    """
    gamma = 1.0 / infectious_period
    
    # Negative Binomial degree distribution to model superspreading
    # Mean degree = 10, Variance = 30 (high overdispersion)
    mean_k = 10.0
    var_k = 30.0
    
    p_nb = mean_k / var_k
    r_nb = mean_k**2 / (var_k - mean_k)
    
    max_k = 150
    ks = np.arange(0, max_k)
    P_k = nbinom.pmf(ks, r_nb, p_nb)
    P_k = P_k / np.sum(P_k)
    
    moment1 = np.sum(ks * P_k)
    moment2 = np.sum(ks**2 * P_k)
    
    # R0 = (tau / (tau + gamma)) * (<K^2> - <K>) / <K>
    x = R0 * moment1 / (moment2 - moment1)
    if x >= 1.0:
        raise ValueError(f"R0={R0} is too high for this degree distribution.")
        
    baseline_tau = x * gamma / (1.0 - x)
    
    # Leaky vaccination
    effective_tau = baseline_tau * (1.0 - coverage * efficacy)
    
    rho = initial_infected / N
    
    def psihat(x_val):
        x_val = np.asarray(x_val)
        return (1-rho) * np.sum(P_k * x_val[..., np.newaxis]**ks, axis=-1)
        
    def psihatPrime(x_val):
        x_val = np.asarray(x_val)
        ks_minus_1 = np.maximum(ks - 1, 0)
        return (1-rho) * np.sum(ks * P_k * x_val[..., np.newaxis]**ks_minus_1, axis=-1)

    phiS0 = 1.0 - rho
    
    t, S, I, R = EoN.EBCM(N, psihat, psihatPrime, effective_tau, gamma, phiS0, 
                          tmax=300, tcount=300)
    
    return R[-1] / N
