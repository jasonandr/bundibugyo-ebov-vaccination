import numpy as np
from plot_rt_calibration_test import estimate_rt_from_incidence

# Exponential growth: exact doubling every generation
# If generation time is ~12 days, then Rt = 2.0.
inc_small = np.array([2, 4, 8, 16, 32, 64, 128], dtype=float)
inc_large = inc_small * 100

rt_small = estimate_rt_from_incidence(inc_small, prior_sd=0.6)
rt_large = estimate_rt_from_incidence(inc_large, prior_sd=0.6)

print("Rt with small incidence:", rt_small[-1])
print("Rt with large incidence:", rt_large[-1])
