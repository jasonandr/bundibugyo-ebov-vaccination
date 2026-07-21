import numpy as np
import matplotlib.pyplot as plt
import datetime
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination

print("Generating Fig 7: The Trial Design Dilemma (Immediate vs Delayed)...")

N = 5000
G = generate_network(N)
tau = calibrate_tau(G, 1.6, 1.0/6.0)

efficacies = np.linspace(0.1, 1.0, 10)
num_runs = 50

imm_cases_mean, imm_cases_std = [], []
del_cases_mean, del_cases_std = [], []
imm_deaths_mean, imm_deaths_std = [], []
del_deaths_mean, del_deaths_std = [], []

for eff in efficacies:
    print(f"Evaluating Efficacy: {eff*100:.0f}%")
    
    imm_c, del_c = [], []
    imm_d, del_d = [], []
    
    for _ in range(num_runs):
        # Immediate Arm (Standard 2 day trace delay)
        res_i, death_i, _ = simulate_ring_vaccination(
            G, baseline_tau=tau, efficacy=eff, tracing_delay=2.0, max_cases=1000
        )
        imm_c.append(res_i * N)
        imm_d.append(death_i * N)
        
        # Delayed Arm (21-day trial delay + standard 2 day trace delay)
        res_d, death_d, _ = simulate_ring_vaccination(
            G, baseline_tau=tau, efficacy=eff, tracing_delay=23.0, max_cases=1000
        )
        del_c.append(res_d * N)
        del_d.append(death_d * N)
        
    imm_cases_mean.append(np.mean(imm_c))
    imm_cases_std.append(np.std(imm_c))
    del_cases_mean.append(np.mean(del_c))
    del_cases_std.append(np.std(del_c))
    
    imm_deaths_mean.append(np.mean(imm_d))
    imm_deaths_std.append(np.std(imm_d))
    del_deaths_mean.append(np.mean(del_d))
    del_deaths_std.append(np.std(del_d))

imm_cases_mean = np.array(imm_cases_mean)
del_cases_mean = np.array(del_cases_mean)
imm_cases_std = np.array(imm_cases_std)
del_cases_std = np.array(del_cases_std)

imm_deaths_mean = np.array(imm_deaths_mean)
del_deaths_mean = np.array(del_deaths_mean)
imm_deaths_std = np.array(imm_deaths_std)
del_deaths_std = np.array(del_deaths_std)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Plot 1: Total Cases
ax1.plot(efficacies * 100, del_cases_mean, 'r-o', linewidth=2, label="Delayed Arm (Day 21)")
ax1.fill_between(efficacies * 100, del_cases_mean - del_cases_std*0.2, del_cases_mean + del_cases_std*0.2, color='red', alpha=0.1)

ax1.plot(efficacies * 100, imm_cases_mean, 'b-o', linewidth=2, label="Immediate Arm (Day 0)")
ax1.fill_between(efficacies * 100, imm_cases_mean - imm_cases_std*0.2, imm_cases_mean + imm_cases_std*0.2, color='blue', alpha=0.1)

ax1.set_title("Cluster Outbreak Size by Trial Arm", fontsize=16, fontweight='bold')
ax1.set_xlabel("True Vaccine Efficacy (%)", fontsize=14)
ax1.set_ylabel("Total Cases per Outbreak Cluster", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Excess Deaths (The Ethical Cost)
excess_deaths = del_deaths_mean - imm_deaths_mean
excess_std = np.sqrt(del_deaths_std**2 + imm_deaths_std**2) * 0.2

ax2.bar(efficacies * 100, excess_deaths, width=6, color='darkorange', alpha=0.8, edgecolor='black', yerr=excess_std, capsize=5)
ax2.axhline(0, color='black', linewidth=1)

ax2.set_title("The Ethical Cost of Delaying Vaccination", fontsize=16, fontweight='bold')
ax2.set_xlabel("True Vaccine Efficacy (%)", fontsize=14)
ax2.set_ylabel("Excess Deaths per Cluster (Due to 21-Day Delay)", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

plt.suptitle("Policy Analysis: Immediate vs. Delayed Trial Design for Unproven BDBV Vaccines", fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/fig7_trial_design_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved {filename}")
