import numpy as np
import matplotlib.pyplot as plt

inc = np.load('results/simulated_incidences.npy')
rt = np.load('results/rt_spaghetti_arrays.npy')

# Replicates that survived to day 60 (i.e. not NaN at day 60)
surviving = ~np.isnan(rt[:, 60])
num_surviving = np.sum(surviving)

mean_inc_all = np.mean(inc, axis=0)
mean_inc_surviving = np.mean(inc[surviving], axis=0)

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(mean_inc_all, color='black', label=f'Mean Incidence (All {len(inc)} reps)')
ax1.plot(mean_inc_surviving, color='blue', label=f'Mean Incidence (Only {num_surviving} surviving reps)')
ax1.set_ylabel("Daily Incidence")
ax1.set_xlabel("Days since May 15")

ax2 = ax1.twinx()
mean_rt_all = np.nanmean(rt, axis=0)
ax2.plot(mean_rt_all, color='red', linestyle='--', label='Mean True Rt (Excludes extinct reps)')
ax2.axhline(1.0, color='gray', linestyle=':')
ax2.set_ylabel("True Rt", color='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Statistical Artifact: Why Mean Incidence Falls while True Rt > 1")
plt.savefig('../figures/survival_artifact.png')
print(f"Created plot with {num_surviving} surviving reps.")
