import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pandas as pd

print("Generating Idea 6: Bivariate Hexbin...")

# Generate synthetic correlated data
np.random.seed(42)
n_points = 5000

# Vaccine Efficacy (0 to 1)
efficacy = np.random.beta(2, 2, n_points)

# Outbreak Size (negatively correlated with efficacy, with high variance at low efficacy)
base_size = 800
cases = base_size * (1 - efficacy) + np.random.exponential(100 + 400 * (1 - efficacy), n_points)
cases = np.clip(cases, 0, 2000)

df = pd.DataFrame({"Vaccine Efficacy": efficacy, "Final Outbreak Size": cases})

sns.set_theme(style="ticks")
g = sns.jointplot(
    data=df,
    x="Vaccine Efficacy", y="Final Outbreak Size",
    kind="hex", color="#4CB391",
    gridsize=30,
    marginal_kws=dict(bins=30, fill=True)
)

g.fig.suptitle("Hexagonal Binning: Efficacy vs Outbreak Size", y=1.03, fontsize=16, fontweight='bold')
g.ax_joint.set_xlabel("Vaccine Efficacy", fontsize=14)
g.ax_joint.set_ylabel("Final Outbreak Size", fontsize=14)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea6_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight')
print(f"Saved {filename}")
