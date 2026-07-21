import numpy as np
import matplotlib.pyplot as plt
import os
import time

def main():
    inc_path = '../results/simulated_incidences.npy'
    if not os.path.exists(inc_path):
        print("Incidences not found!")
        return
        
    incidences = np.load(inc_path)
    mean_inc = np.mean(incidences, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(incidences)):
        ax.plot(incidences[i], color='gray', alpha=0.15)
        
    ax.plot(mean_inc, color='#C0392B', lw=3, label='Mean Simulated Daily Incidence')
    ax.set_title("Simulated Daily Incidence (No Empirical Data)", fontsize=16, pad=15)
    ax.set_ylabel("Incident Cases", fontsize=14)
    ax.set_xlabel("Days since start", fontsize=14)
    ax.axvline(120, color='blue', linestyle='--', label="~July 1 (Day 120)")
    ax.legend(frameon=False, fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    timestamp = int(time.time())
    img_path = f"../figures/simulated_cases_{timestamp}.png"
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved to {img_path}")
    
if __name__ == '__main__':
    main()
