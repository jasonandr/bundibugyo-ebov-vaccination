import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

timestamp = int(time.time())
OUT_DIR = Path("figures/polished")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def setup_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

setup_style()

def plot_fig2_forest_psa():
    df_psa = pd.read_csv("../data_and_results/psa_summary_results.csv")
    
    label_map = {
        "vax_base_ops": ("Ring Vaccination (R=1)", "vs Base Operations", "#2F6F9F"),
        "no_vax_enh_ops": ("Enhanced Operations Alone", "vs Base Operations", "#4A5561"),
        "vax_enh_ops": ("Enhanced Operations + Ring Vax", "vs Base Operations", "#2F6F9F"),
        "incremental_ring_vax": ("Incremental Ring Vax", "vs Enhanced Operations", "#2F6F9F"),
        "comm_base_20": ("Community Vaccination 20%", "vs Base Operations", "#1F8E83"),
        "comm_base_40": ("Community Vaccination 40%", "vs Base Operations", "#1F8E83"),
        "comm_base_60": ("Community Vaccination 60%", "vs Base Operations", "#1F8E83"),
        "comm_base_80": ("Community Vaccination 80%", "vs Base Operations", "#1F8E83"),
        "incremental_comm_vax_20": ("Incremental Comm Vax 20%", "vs Enhanced Operations", "#1F8E83"),
    }
    
    order = [
        "vax_base_ops", "no_vax_enh_ops", "vax_enh_ops", "incremental_ring_vax",
        "comm_base_20", "comm_base_40", "comm_base_60", "comm_base_80", "incremental_comm_vax_20"
    ]
    
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    
    y_positions = np.arange(len(order))[::-1]
    
    for i, s_key in enumerate(order):
        row = df_psa[df_psa['scenario'] == s_key]
        if row.empty:
            continue
        row = row.iloc[0]
        y = y_positions[i]
        
        lbl, sublbl, color = label_map[s_key]
        med = row['median_deaths_averted_pct']
        ui_low = row['psa_ui_low_95']
        ui_high = row['psa_ui_high_95']
        iqr_low = row['iqr_low_25']
        iqr_high = row['iqr_high_75']
        
        # 95% UI line (thin)
        ax.plot([ui_low, ui_high], [y, y], color=color, linewidth=1.5, alpha=0.6, zorder=2)
        # IQR line (thick)
        ax.plot([iqr_low, iqr_high], [y, y], color=color, linewidth=3.5, alpha=0.9, zorder=3)
        # Median marker
        ax.plot(med, y, 'o', color='white', markeredgecolor=color, markeredgewidth=2.0, markersize=6.5, zorder=4)
        
        # Label on left
        ax.text(-5.0, y, f"{lbl}\n({sublbl})", ha='right', va='center', fontsize=7.5, color='#111827')
        # Value text on right
        val_str = f"{med:.1f}% [{ui_low:.1f}% – {ui_high:.1f}%]"
        ax.text(102.0, y, val_str, ha='left', va='center', fontsize=7.5, fontweight='bold', color='#111827')

    ax.axvline(0, color='#6B7280', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_xlim(-5, 105)
    ax.set_xlabel("Percent Cumulative Deaths Averted (Median, IQR, and 95% PSA Uncertainty Interval)", fontsize=8.5, fontweight='bold')
    ax.set_title("Figure 2: Impact of Un-confounded Strategies with 90-Day Joint Parameter Uncertainty (PSA)", fontsize=9.5, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig_path = f"/Users/jasonandrews/.gemini/antigravity-ide/brain/b92785c3-f511-471c-a5cd-d92f3cf65e7e/fig2_forest_psa_unconfounded_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 Forest plot to {fig_path}")
    return fig_path

if __name__ == '__main__':
    fig2_p = plot_fig2_forest_psa()
