import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import os

from paths import result_path

raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

baseline_df = raw[(raw["scenario"] == "fig2_base") & (raw["level"] == "no_vax")].copy()
r1_df = raw[(raw["scenario"] == "fig2_base") & (raw["level"] == "radius1")].copy()
r2_df = raw[(raw["scenario"] == "fig2_base") & (raw["level"] == "radius2")].copy()

for df in [baseline_df, r1_df, r2_df]:
    df["cases"] = df["cases_percent"] / 100.0 * df["population_size"]

baseline_df["Strategy"] = "No Vaccination"
r1_df["Strategy"] = "Radius 1 (40% VE)"
r2_df["Strategy"] = "Radius 2 (40% VE)"

combined = pd.concat([baseline_df, r1_df, r2_df])

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

palette = {
    "No Vaccination": "#bdc3c7",
    "Radius 1 (40% VE)": "#2c3e50",
    "Radius 2 (40% VE)": "#e67e22"
}

sns.ecdfplot(data=combined, x="cases", hue="Strategy", palette=palette, ax=ax, linewidth=2.5)

ax.set_xlabel("Total Outbreak Size (Cases)", fontsize=12)
ax.set_ylabel("Cumulative Probability (Containment)", fontsize=12)
ax.set_title("Probability of Containment by Strategy", fontsize=14, fontweight='bold', loc='left')
ax.set_xlim(0, 150)
ax.grid(True, linestyle='--', alpha=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
png_filename = f"figures/fig_cdf_{timestamp}.png"
pdf_filename = f"figures/fig_cdf_{timestamp}.pdf"

plt.tight_layout()
plt.savefig(png_filename, bbox_inches='tight')
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
print(f"Saved CDF plot to {png_filename}")
