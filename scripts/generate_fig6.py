import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from scipy.ndimage import gaussian_filter

XX = np.load('data/fig6_XX.npy') # VE_D
YY = np.load('data/fig6_YY.npy') # VE_M|D
res = np.load('data/fig6_res.npy')
res = gaussian_filter(res, sigma=1.0)
baseline = np.max(res)
avert = np.maximum(0, (baseline - res) / baseline * 100)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

X, Y = np.meshgrid(XX * 100, YY * 100)
contour = ax.contourf(X, Y, avert, levels=15, cmap='plasma')
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('% Total Deaths Averted', fontsize=12)

ax.contour(X, Y, avert, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Plot a star at our baseline 30/50
ax.scatter(30, 50, color='lime', marker='*', s=200, edgecolor='black', zorder=5, label='Baseline (30% Susc, 50% Ther)')
ax.legend(loc='lower right')

ax.set_title("Figure 6: Efficacy vs Therapeutic Rescue", fontsize=14, fontweight='bold')
ax.set_xlabel("Protection against Disease ($VE_{D}$) (%)", fontsize=12)
ax.set_ylabel("Protection against Mortality conditional on Disease ($VE_{M|D}$) (%)", fontsize=12)

plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
png_filename = f"figures/fig6_ve_matrix_{timestamp}.png"
pdf_filename = f"figures/fig6_ve_matrix_{timestamp}.pdf"

plt.savefig(png_filename, bbox_inches='tight')
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
print(f"Saved Figure 6 to {png_filename} and .pdf")

walkthrough_path = "figures/walkthrough.md"
import re
if os.path.exists(walkthrough_path):
    with open(walkthrough_path, "a") as f:
        f.write(f"\n\n![Figure 6]({png_filename})\n")
