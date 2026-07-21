import numpy as np

z = np.load("../data_and_results/new_spaghetti_chunks/chunk_1.npz", allow_pickle=True)
base_no_vax = z["base_no_vax"]
print("Shape:", base_no_vax.shape)
print("Type:", type(base_no_vax))
print("First element shape:", base_no_vax[0].shape)
