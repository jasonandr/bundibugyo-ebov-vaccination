import numpy as np
import matplotlib.pyplot as plt
from ebola_stochastic_ring import generate_network

def sim_offspring(pooled=False):
    G = generate_network(10000, household_mean=5.2, community_mean=5.0, community_variance=25.0)
    state = np.zeros(len(G.nodes), dtype=int)
    # 0=S, 1=I
    
    # Let's just do a 1-generation test. We infect 1000 random people.
    seed_nodes = np.random.choice(len(G.nodes), 1000, replace=False)
    for n in seed_nodes: state[n] = 1
    
    offspring_counts = {n: 0 for n in seed_nodes}
    target_rt = 2.5
    
    if not pooled:
        for node in seed_nodes:
            n_infect = int(np.floor(target_rt))
            if np.random.rand() < (target_rt - n_infect):
                n_infect += 1
            susceptible = [n for n in G.neighbors(node) if state[n] == 0]
            actual = min(n_infect, len(susceptible))
            if actual > 0:
                chosen = np.random.choice(susceptible, actual, replace=False)
                for c in chosen: state[c] = 1
                offspring_counts[node] += actual
    else:
        expected = len(seed_nodes) * target_rt
        target = int(np.floor(expected))
        if np.random.rand() < (expected - target): target += 1
        
        pool = []
        for node in seed_nodes:
            for neighbor in G.neighbors(node):
                if state[neighbor] == 0:
                    pool.append((node, neighbor))
        
        np.random.shuffle(pool)
        actual = 0
        for source, target_node in pool:
            if state[target_node] == 0:
                state[target_node] = 1
                offspring_counts[source] += 1
                actual += 1
                if actual >= target: break
                
    return list(offspring_counts.values())

print("Running Individual Targeting...")
dist_indiv = sim_offspring(pooled=False)
print("Running Cohort Pooling...")
dist_pooled = sim_offspring(pooled=True)

plt.figure(figsize=(10, 5))
plt.hist(dist_indiv, bins=range(20), alpha=0.5, label='Individual Target (Test 1)', density=True)
plt.hist(dist_pooled, bins=range(20), alpha=0.5, label='Cohort Pooled (Test 2)', density=True)
plt.title('Offspring Distribution (Target Rt = 2.5)')
plt.xlabel('Secondary Infections per Person')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/offspring_comparison.png', dpi=300)
