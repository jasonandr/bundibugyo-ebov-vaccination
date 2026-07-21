import numpy as np
import matplotlib.pyplot as plt
import datetime

print("Generating Idea 5: Adjacency Matrix Fingerprint...")

N_cases = 300

# Create a baseline block-diagonal matrix (household clusters)
base_matrix = np.zeros((N_cases, N_cases))
idx = 0
while idx < N_cases:
    hh_size = np.random.poisson(4) + 1
    if idx + hh_size > N_cases:
        hh_size = N_cases - idx
    base_matrix[idx:idx+hh_size, idx:idx+hh_size] = 1
    idx += hh_size

# Add baseline community spread (sparse off-diagonal)
sparse_base = np.random.rand(N_cases, N_cases) < 0.005
base_matrix[sparse_base] = 1

# Scenario B: Risk Compensation (dense off-diagonal)
risk_matrix = np.copy(base_matrix)
sparse_risk = np.random.rand(N_cases, N_cases) < 0.03 # 6x more community spread
risk_matrix[sparse_risk] = 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

ax1.matshow(base_matrix, cmap='Blues')
ax1.set_title("Standard Outbreak\n(Transmission concentrated in households)", fontsize=14, pad=20)
ax1.set_xlabel("Infected Individual ID (Sorted by Time)", fontsize=12)
ax1.set_ylabel("Infected Individual ID", fontsize=12)

ax2.matshow(risk_matrix, cmap='Reds')
ax2.set_title("Risk Compensation Paradox\n(Breakdown of ring containment; rampant community spread)", fontsize=14, pad=20)
ax2.set_xlabel("Infected Individual ID (Sorted by Time)", fontsize=12)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea5_{timestamp}.png"
plt.tight_layout()
plt.savefig(filename, bbox_inches='tight')
print(f"Saved {filename}")
