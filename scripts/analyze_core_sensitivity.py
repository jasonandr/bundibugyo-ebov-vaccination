import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
import datetime

print("Generating 5000-node network for LHS sensitivity sweep...")
G = generate_network(5000)

print("Calibrating baseline transmission probability (tau) for R0=1.6...")
tau = calibrate_tau(G, target_R0=1.6, gamma=1.0/6.0, num_trials=50)
print(f"Calibrated tau: {tau:.4f}")

num_samples = 400

# Define parameter bounds
bounds = {
    "Case Detection Rate": (0.2, 1.0),
    "Tracing Coverage": (0.2, 1.0),
    "Vaccine Acceptability": (0.4, 1.0),
    "Detection Delay (days)": (2.0, 8.0),
    "Base CFR": (0.4, 0.75),
    "Vaccinated CFR": (0.15, 0.35)
}

param_names = list(bounds.keys())
X = np.zeros((num_samples, len(param_names)))

# Generate Random Uniform Samples (approx LHS)
for i, name in enumerate(param_names):
    low, high = bounds[name]
    X[:, i] = np.random.uniform(low, high, num_samples)

results_cases = np.zeros(num_samples)
results_deaths = np.zeros(num_samples)

print(f"Starting {num_samples} simulations for sensitivity analysis...")
for i in range(num_samples):
    res, deaths, vaccines = simulate_ring_vaccination(
        G, 
        rt_array=None, 
        baseline_tau=tau, 
        incubation_period=8.5, 
        infectious_period=6.0, 
        uptake=0.0, # Overridden by the new parameters
        efficacy=0.5, # Fixed intermediate efficacy
        reporting_rate=X[i, 0],
        tracing_coverage=X[i, 1],
        vaccine_acceptability=X[i, 2],
        detection_delay=X[i, 3],
        base_CFR=X[i, 4],
        vax_CFR=X[i, 5],
        max_sim_time=300
    )
    results_cases[i] = res * 100
    results_deaths[i] = deaths * 100
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{num_samples} simulations completed.")

# Calculate Spearman Rank Correlation Coefficients against Total Deaths
correlations = []
for i, name in enumerate(param_names):
    corr, _ = spearmanr(X[:, i], results_deaths)
    correlations.append((name, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=False)

labels = [f"{x[0]} [{bounds[x[0]][0]}-{bounds[x[0]][1]}]" for x in correlations]
values = [x[1] for x in correlations]

# Generate Tornado Plot
plt.figure(figsize=(10, 6), dpi=150)
# Sleeker colors: Coral/Orange for positive correlation, Deep Indigo/Blue for negative
colors = ['#FF6F61' if v > 0 else '#4A90E2' for v in values]

y_pos = np.arange(len(labels))
plt.barh(y_pos, values, color=colors, edgecolor='black')
plt.yticks(y_pos, labels, fontsize=12)
plt.axvline(x=0, color='black', linewidth=1.5)

plt.xlabel("Spearman Rank Correlation Coefficient (PRCC proxy)", fontsize=12)
plt.title("Tornado Diagram: Sensitivity of Total Deaths to Core Parameters", fontsize=14, fontweight='bold')

# Annotations
for i, v in enumerate(values):
    plt.text(v + (0.02 if v > 0 else -0.02), i, f"{v:.2f}", 
             va='center', ha='left' if v > 0 else 'right', fontsize=10, fontweight='bold')

plt.xlim(-1.0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/tornado_core_sensitivity_{timestamp}.png"
plt.savefig(filename)
print(f"Saved plot to {filename}")

# Automatic Markdown File update
import re
walkthrough_path = "figures/walkthrough.md"
with open(walkthrough_path, "r") as f:
    content = f.read()

section = f"""
## 8. Tornado Diagram: Core Intervention Logistical Sensitivity

We generated a multidimensional sensitivity analysis across the core operational parameters you specified (Case Detection, Tracing Coverage, Vaccine Acceptability, Delays, and CFR bounds).

Using 400 Monte Carlo runs and calculating the Spearman Rank Correlation Coefficients, we can visualize exactly which parameter exerts the strongest "pull" on the final death toll.

> [!TIP]
> **Interpreting the Tornado Plot**: 
> *   **Blue Bars (Negative Correlation)**: Increasing this parameter *decreases* deaths. (e.g., Higher Case Detection strongly mitigates the outbreak).
> *   **Coral Bars (Positive Correlation)**: Increasing this parameter *increases* deaths. (e.g., Higher Base CFR or Detection Delays).
> *   The length of the bar indicates the sheer strength of the effect.

![Tornado Diagram Core Sensitivity]({filename})
"""

if "## 8. Tornado Diagram:" in content:
    pattern = r"!\[Tornado Diagram Core Sensitivity\]\(.*?\)"
    new_content = re.sub(pattern, f"![Tornado Diagram Core Sensitivity]({filename})", content)
    with open(walkthrough_path, "w") as f:
        f.write(new_content)
else:
    with open(walkthrough_path, "a") as f:
        f.write(section)

print("Walkthrough updated successfully.")
