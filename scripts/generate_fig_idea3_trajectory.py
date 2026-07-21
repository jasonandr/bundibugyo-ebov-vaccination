import matplotlib.pyplot as plt
import numpy as np
import datetime
from ebola_stochastic_ring import generate_network, simulate_ring_vaccination

print("Generating Idea 3: Trajectory Spaghetti Plot...")

N = 5000
G = generate_network(N)
num_sims = 30
max_t = 150

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

print("Running Radius 1 simulations...")
for _ in range(num_sims):
    inc = simulate_ring_vaccination(G, ring_radius=1, efficacy=0.7, max_sim_time=max_t, return_time_series=True, baseline_tau=0.1)
    ax.plot(inc[:max_t], color='blue', alpha=0.15, linewidth=1.5)

print("Running Radius 2 simulations...")
for _ in range(num_sims):
    inc = simulate_ring_vaccination(G, ring_radius=2, efficacy=0.7, max_sim_time=max_t, return_time_series=True, baseline_tau=0.1)
    ax.plot(inc[:max_t], color='red', alpha=0.15, linewidth=1.5)

# Add medians
ax.plot([], [], color='blue', label='Radius 1 Trajectories', linewidth=2)
ax.plot([], [], color='red', label='Radius 2 Trajectories', linewidth=2)

ax.set_title("Outbreak Velocity: The Danger of Radius 2 Oscillations", fontsize=16, fontweight='bold')
ax.set_xlabel("Days Since Outbreak Start", fontsize=14)
ax.set_ylabel("Daily New Cases (Incidence)", fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea3_{timestamp}.png"
plt.tight_layout()
plt.savefig(filename, bbox_inches='tight')
print(f"Saved {filename}")
