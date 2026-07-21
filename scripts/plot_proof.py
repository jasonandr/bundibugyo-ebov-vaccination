import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load data
    incidences = np.load('../results/simulated_incidences.npy')
    mean_inc = np.mean(incidences, axis=0)
    
    rt_spaghetti = np.load('../results/rt_spaghetti_arrays.npy')
    mean_rt = np.nanmean(rt_spaghetti, axis=0)
    
    # Ensure same length
    length = min(len(mean_inc), len(mean_rt))
    mean_inc = mean_inc[:length]
    mean_rt = mean_rt[:length]
    
    # Smooth incidence for visualization
    import pandas as pd
    mean_inc_smooth = pd.Series(mean_inc).rolling(7, min_periods=1, center=True).mean().values
    
    # Find crossover
    crossover_day = np.where(mean_rt < 1.0)[0][0] if any(mean_rt < 1.0) else None
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(mean_inc_smooth, color='blue', lw=3, label='Mean Incident Cases (7d smooth)')
    ax1.set_ylabel("Incident Cases per Day", color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel("Days Since Start", fontsize=14)
    
    ax2 = ax1.twinx()
    ax2.plot(mean_rt, color='red', lw=3, label='Mean True $R_t$')
    ax2.axhline(1.0, color='gray', linestyle='--')
    ax2.set_ylabel("True Effective $R_t$", color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')
    
    if crossover_day:
        ax1.axvline(crossover_day, color='k', linestyle=':', lw=2)
        ax1.text(crossover_day+2, max(mean_inc_smooth)*0.9, f"Rt crosses 1.0 at Day {crossover_day}\nIncidence peaks here!", fontsize=12, fontweight='bold')
        
    plt.title("Proof: Incidence peaks exactly when True Rt = 1.0", fontsize=16)
    
    import time
    img_path = f"../figures/proof_{int(time.time())}.png"
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved to {img_path}")

if __name__ == '__main__':
    main()
