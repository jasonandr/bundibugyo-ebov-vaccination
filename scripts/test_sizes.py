import json
from ebola_stochastic_ring import calibrate_tau, generate_network, simulate_ring_vaccination

def main():
    G = generate_network(10000)
    tau = calibrate_tau(G, 7.22, 1.0/6.0)
    
    # Run 10 times and get average size
    sizes_base = []
    sizes_enh = []
    import numpy as np
    
    base_r = [0.3] * 91
    enh_r = np.linspace(0.3, 0.7, 15).tolist() + [0.7] * 76
    enh_t = np.linspace(0.3, 0.8, 15).tolist() + [0.8] * 76
    
    for i in range(100):
        c, _, _, _ = simulate_ring_vaccination(
            G, baseline_tau=tau, reporting_rate=base_r, tracing_coverage=[0.3]*91,
            vaccine_effect=0, base_CFR=0.5, engine='cpp', seed=i, initial_infected=5, max_sim_time=90
        )
        sizes_base.append(c)
        
        c2, _, _, _ = simulate_ring_vaccination(
            G, baseline_tau=tau, reporting_rate=enh_r, tracing_coverage=enh_t,
            vaccine_effect=0, base_CFR=0.5, engine='cpp', seed=i, initial_infected=5, max_sim_time=90
        )
        sizes_enh.append(c2)
        
    print("Base mean size:", np.mean(sizes_base))
    print("Enh mean size:", np.mean(sizes_enh))
    print("Reduction:", (np.mean(sizes_base) - np.mean(sizes_enh)) / np.mean(sizes_base))
    
if __name__ == '__main__':
    main()
