import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from paths import figure_path

def main():
    # Load required data
    if not os.path.exists('../results/epinow_rt.csv'):
        print("EpiNow2 Rt results not found. Run estimate_rt.py first.")
        return
        
    rt_df = pd.read_csv('../results/epinow_rt.csv')
    rt_df['date'] = pd.to_datetime(rt_df['date'])
    
    # Extract baseline true Rt arrays
    spaghetti_path = '../results/rt_spaghetti_arrays.npy'
    if not os.path.exists(spaghetti_path):
        print("Spaghetti arrays not found. Run generate_final_outputs.py first.")
        return
        
    all_rt_arrays = np.load(spaghetti_path)
    mean_rt = np.nanmean(all_rt_arrays, axis=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_sim_time = min(len(rt_df), len(mean_rt))
    
    # Plot true Rt spaghetti
    for i in range(min(50, len(all_rt_arrays))):
        ax.plot(rt_df['date'][:max_sim_time], all_rt_arrays[i, :max_sim_time], color='#E74C3C', alpha=0.1, lw=1)
        
    ax.plot(rt_df['date'][:max_sim_time], mean_rt[:max_sim_time], color='#C0392B', lw=3, label='Simulated True $R_t$ (Mean)')
    
    # Plot EpiNow2 Rt
    ax.plot(rt_df['date'][:max_sim_time], rt_df['median'][:max_sim_time], color='#2980B9', lw=2.5, linestyle='--', label='Empirical Forcing (EpiNow2 Median $R_t$)')
    ax.fill_between(rt_df['date'][:max_sim_time], rt_df['lower_90'][:max_sim_time], rt_df['upper_90'][:max_sim_time], color='#2980B9', alpha=0.2, label='90% CrI')
    
    ax.axhline(1.0, color='k', linestyle=':', lw=1.5, alpha=0.5)
    ax.set_ylabel(r"Effective $R_t$", fontsize=14)
    ax.set_xlabel("Date (2026)", fontsize=14)
    ax.set_title("Appendix: Simulated vs. Empirical Effective Reproduction Number", fontsize=16, pad=15)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', frameon=False, fontsize=12)
    
    plt.tight_layout()
    timestamp = int(time.time())
    
    os.makedirs(figure_path("appendix"), exist_ok=True)
    img_path = figure_path("appendix") / f"appendix_rt_spaghetti_{timestamp}.png"
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved Appendix Rt Spaghetti to {img_path}")
    
    # Save to dropbox
    dropbox_dir = "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/Figures_v43_high_res"
    if os.path.exists(dropbox_dir):
        plt.savefig(f"{dropbox_dir}/Appendix_rt_spaghetti.png", dpi=300, facecolor='white')
        plt.savefig(f"{dropbox_dir}/Appendix_rt_spaghetti.pdf", dpi=300, facecolor='white', format='pdf')

if __name__ == "__main__":
    main()
