import numpy as np

# A quick mock of the variance logic
expected = 100
trajs = []
for i in range(50):
    val = np.floor(expected)
    if np.random.rand() < (expected - val):
        val += 1
    trajs.append(val)
print(f"Max variation at a step: {np.max(trajs) - np.min(trajs)}")
