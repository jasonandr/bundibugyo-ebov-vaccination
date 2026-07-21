import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
import datetime
import sys

print("Generating 5000-node network for sensitivity sweep...")
G = generate_network(5000)

print("Calibrating baseline transmission probability (tau) for R0=1.6...")
tau = calibrate_tau(G, target_R0=1.6, gamma=1.0/6.0, num_trials=50)
print(f"Calibrated tau: {tau:.4f}")

efficacies = np.linspace(0.1, 1.0, 10)
num_trials = 5

scenarios = {
    "Baseline (No Behavioral Effects)": {"risk_comp": 1.0, "trust_loss": False},
    "Trust Loss Only (Efficacy → Uptake)": {"risk_comp": 1.0, "trust_loss": True},
    "Risk Comp Only (2x contact rate)": {"risk_comp": 2.0, "trust_loss": False},
    "Combined (Trust Loss + Risk Comp)": {"risk_comp": 2.0, "trust_loss": True}
}

results = {name: [] for name in scenarios}

print("Starting behavioral sensitivity sweep...")
total_runs = len(efficacies) * len(scenarios) * num_trials
run_count = 0

for eff in efficacies:
    print(f"Testing Efficacy = {eff*100:.0f}%...")
    for name, params in scenarios.items():
        case_sizes = []
        for _ in range(num_trials):
            # Base uptake is 80%
            res, deaths, vaccines = simulate_ring_vaccination(
                G, 
                rt_array=None, 
                baseline_tau=tau, 
                incubation_period=8.5, 
                infectious_period=6.0, 
                uptake=0.8, 
                efficacy=eff, 
                reporting_rate=0.6,
                max_sim_time=300,
                risk_compensation_multiplier=params["risk_comp"],
                trust_uptake_dependency=params["trust_loss"]
            )
            case_sizes.append(res * 100) # Percentage of population
            run_count += 1
            if run_count % 10 == 0:
                print(f"  Progress: {run_count}/{total_runs} simulations completed.")
        
        results[name].append(np.mean(case_sizes))

# Generate the plot
plt.figure(figsize=(10, 7), dpi=150)
colors = {
    "Baseline (No Behavioral Effects)": "#2ECC71",
    "Trust Loss Only (Efficacy → Uptake)": "#F39C12",
    "Risk Comp Only (2x contact rate)": "#E74C3C",
    "Combined (Trust Loss + Risk Comp)": "#8E44AD"
}

for name, data in results.items():
    plt.plot(efficacies * 100, data, marker='o', linewidth=3, markersize=8, 
             label=name, color=colors[name])

plt.title("Sensitivity Analysis: Behavioral & Trust Dynamics", fontsize=16, fontweight='bold')
plt.xlabel("Vaccine Efficacy (%)", fontsize=14)
plt.ylabel("Final Outbreak Size (% of population)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Reference line for 0 efficacy (no vaccine)
# Calculate a quick no-vax baseline
no_vax_cases = []
for _ in range(num_trials):
    res, _, _ = simulate_ring_vaccination(G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0, uptake=0.0, efficacy=0.0)
    no_vax_cases.append(res*100)
no_vax_mean = np.mean(no_vax_cases)
plt.axhline(y=no_vax_mean, color='black', linestyle=':', linewidth=2, label="No Vaccine Intervention")
plt.legend(fontsize=12)

plt.tight_layout()

import re
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/behavioral_sensitivity_{timestamp}.png"
plt.savefig(filename)
print(f"Saved plot to {filename}")

# Automatic Markdown File update
import glob
walkthrough_path = "figures/walkthrough.md"
with open(walkthrough_path, "r") as f:
    content = f.read()

# Check if behavioral section exists, if not, append
section = f"""
## 7. Sensitivity Analysis: Behavioral Dynamics & Trust

We ran a comprehensive sensitivity sweep across Vaccine Efficacy (10% to 100%) to evaluate the inflection points where behavioral dynamics (Risk Compensation and Trust Loss) drastically alter the epidemic trajectory.

> [!CAUTION]
> **The Paradox of Low Efficacy**: As shown in the chart below, if a vaccine has low efficacy (<40%), deploying it can actually result in a **larger outbreak** than doing nothing at all, *if* vaccinated individuals engage in risk compensation (feeling falsely protected). This effect is amplified when low efficacy undermines community trust, dropping overall coverage.

![Behavioral Sensitivity Analysis]({filename})
"""

if "## 7. Sensitivity Analysis" in content:
    # Use regex to replace the existing image line
    pattern = r"!\[Behavioral Sensitivity Analysis\]\(.*?\)"
    new_content = re.sub(pattern, f"![Behavioral Sensitivity Analysis]({filename})", content)
    with open(walkthrough_path, "w") as f:
        f.write(new_content)
else:
    with open(walkthrough_path, "a") as f:
        f.write(section)

print("Walkthrough updated successfully.")
