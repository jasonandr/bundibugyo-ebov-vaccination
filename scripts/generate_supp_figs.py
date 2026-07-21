import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import os
import pickle
from scipy.ndimage import gaussian_filter

XX = np.load('data/fig6_XX.npy') # VE_D
YY = np.load('data/fig6_YY.npy') # VE_M|D

# Generate S1 and S2 (Matrices)
for year, base_cfr in [("2007", 0.25), ("2012", 0.51)]:
    res = np.load(f'data/supp_matrix_{year}_res.npy')
    res = gaussian_filter(res, sigma=1.0)
    baseline = np.max(res)
    avert = np.maximum(0, (baseline - res) / baseline * 100)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    X, Y = np.meshgrid(XX * 100, YY * 100)
    contour = ax.contourf(X, Y, avert, levels=15, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('% Total Deaths Averted', fontsize=12)

    ax.contour(X, Y, avert, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    # Plot baseline 30/50
    ax.scatter(30, 50, color='lime', marker='*', s=200, edgecolor='black', zorder=5, label='Baseline (30% Susc, 50% Ther)')
    ax.legend(loc='lower right')

    ax.set_title(f"Figure S{'1' if year=='2007' else '2'}: {year} Outbreak Efficacy Matrix\n(Base CFR: {int(base_cfr*100)}%)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Protection against Disease ($VE_{D}$) (%)", fontsize=12)
    ax.set_ylabel("Protection against Mortality conditional on Disease ($VE_{M|D}$) (%)", fontsize=12)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    figname = f"figS{'1' if year=='2007' else '2'}_matrix_{year}_{timestamp}.png"
    filepath = f"figures/{figname}"
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

# Generate S3 and S4 (Violins)
for year in ["2007", "2012"]:
    with open(f'data/supp_violin_{year}_res.pkl', 'rb') as f:
        res = pickle.load(f)
    
    detect_rates = [0.2, 0.4, 0.6]
    labels_d = ["20% Detect", "40% Detect", "60% Detect"]
    
    # Process data into dataframe
    data = []
    idx = 0
    VIOLIN_TRIALS = 200
    for r in [1, 2]:
        radius_label = "Radius 1 (Direct)" if r == 1 else "Radius 2 (Extended)"
        for d_idx, d in enumerate(detect_rates):
            chunk = res[idx:idx+VIOLIN_TRIALS]
            chunk = [x for x in chunk if x is not None]
            for (deaths, vax) in chunk:
                data.append({
                    "Radius": radius_label,
                    "Detection": labels_d[d_idx],
                    "Total Deaths": deaths
                })
            idx += VIOLIN_TRIALS
            
    df = pd.DataFrame(data)
    
    # Calculate baseline max for standardizing aversion
    baseline_deaths = df['Total Deaths'].max()
    df['Deaths Averted (%)'] = np.maximum(0, (baseline_deaths - df['Total Deaths']) / baseline_deaths * 100)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.violinplot(data=df, x="Detection", y="Deaths Averted (%)", hue="Radius",
                   split=False, inner="quartile", palette=["#2c3e50", "#d35400"], alpha=0.3, ax=ax)
    sns.stripplot(data=df, x="Detection", y="Deaths Averted (%)", hue="Radius",
                  dodge=True, size=2, color=".3", alpha=0.3, ax=ax)
                  
    ax.set_title(f"Figure S{'3' if year=='2007' else '4'}: {year} Outbreak Operational Bandwidth", fontsize=14, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Ring Strategy", loc='lower right')
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    figname = f"figS{'3' if year=='2007' else '4'}_violin_{year}_{timestamp}.png"
    filepath = f"figures/{figname}"
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

print("Supplemental Figures Generated!")
