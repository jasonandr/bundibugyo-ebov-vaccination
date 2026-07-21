import pandas as pd
import numpy as np
import joypy
import matplotlib.pyplot as plt
import datetime

print("Generating Idea 4: Stochastic Ridgeline Plot...")

# Generate synthetic time-series data for stochastic outbreaks
np.random.seed(42)
n_runs = 200
weeks = list(range(1, 13))

data = []
for run in range(n_runs):
    # Simulated outbreak curve (gamma-like)
    peak_week = np.random.normal(5, 1.5)
    peak_cases = np.random.exponential(100) + 20
    for w in weeks:
        cases = max(0, peak_cases * np.exp(-0.5 * ((w - peak_week) / 1.5)**2))
        # Add stochastic noise
        cases = np.random.poisson(cases)
        data.append({"Week": f"Week {w}", "Cases": cases, "Scenario": "Radius 2 (Delayed Containment)"})

df = pd.DataFrame(data)

# Create the joyplot
fig, axes = joypy.joyplot(df, by="Week", column="Cases", 
                          figsize=(10, 8), 
                          colormap=plt.cm.OrRd,
                          alpha=0.7, 
                          linewidth=1,
                          x_range=[0, 300])

plt.title("Evolution of Outbreak Uncertainty (Stochastic Density over Time)", fontsize=16, fontweight='bold')
plt.xlabel("Active Cases", fontsize=14)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea4_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight')
print(f"Saved {filename}")
