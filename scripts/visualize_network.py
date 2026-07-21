import networkx as nx
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random

print("Generating 10,000 node Barabasi-Albert scale-free network...")
G = nx.barabasi_albert_graph(10000, 2)

print("Simulating an outbreak cluster...")
cases = set()
# Pick a high-degree node as the index case
seed = random.choice([n for n, d in G.degree() if d > 20])
cases.add(seed)
frontier = list(G.neighbors(seed))

# Simulate spread until we have ~400 cases (peak outbreak size)
while len(cases) < 400 and frontier:
    curr = frontier.pop(0)
    if curr not in cases:
        cases.add(curr)
        for neighbor in G.neighbors(curr):
            if neighbor not in cases and random.random() < 0.4:
                frontier.append(neighbor)

cases_list = list(cases)
healthy_list = [n for n in G.nodes() if n not in cases]

print("Computing spring layout...")
pos = nx.spring_layout(G, k=0.1, iterations=25, seed=42)

print("Plotting network on white background...")
fig, ax = plt.subplots(figsize=(16, 16), facecolor='white', dpi=300)
ax.set_facecolor('white')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.03, edge_color='#AAAAAA', ax=ax)

# Draw healthy nodes (gray, small)
nx.draw_networkx_nodes(G, pos, nodelist=healthy_list, node_size=2, node_color='#CCCCCC', alpha=0.8, ax=ax)

# Draw case nodes (red, larger, on top)
nx.draw_networkx_nodes(G, pos, nodelist=cases_list, node_size=15, node_color='red', alpha=1.0, ax=ax)

# Zoom in by setting axis limits to the core 95% of positions
x_vals = [p[0] for p in pos.values()]
y_vals = [p[1] for p in pos.values()]
ax.set_xlim(np.percentile(x_vals, 2), np.percentile(x_vals, 98))
ax.set_ylim(np.percentile(y_vals, 2), np.percentile(y_vals, 98))

plt.axis('off')

# Save with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/network_visualization_{timestamp}.png"

plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
print(f"Saved visualization to: {filename}")
