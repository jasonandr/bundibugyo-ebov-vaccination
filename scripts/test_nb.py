import numpy as np

mean_k = 5.0
var_k = 25.0
p_nb = mean_k / var_k
r_nb = mean_k**2 / (var_k - mean_k)
ks = np.random.negative_binomial(r_nb, p_nb, 100000)

print(f"Mean: {np.mean(ks):.2f}")
print(f"Median: {np.median(ks):.2f}")
print(f"% with < 4 community contacts: {np.mean(ks < 4)*100:.2f}%")
print(f"% with < 6 community contacts: {np.mean(ks < 6)*100:.2f}%")
