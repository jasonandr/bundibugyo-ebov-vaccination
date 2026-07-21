import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime
import itertools
import heapq

# Import the user's actual network generation
from ebola_stochastic_ring import generate_network

print("Generating 10,000 node TWO-LAYER network (Household cliques + Negative Binomial Community spread)...")
G = generate_network(10000)

print("Running actual SEIR simulation to extract realistic case cluster at peak...")
# Simplified version of the SEIR model just to get the structural cluster
status = {n: 'S' for n in G.nodes()}
queue = []
counter = itertools.count()

def add_event(t, event_type, target):
    heapq.heappush(queue, (t, next(counter), event_type, target))

initial_nodes = np.random.choice(G.nodes(), size=5, replace=False)
for n in initial_nodes:
    add_event(0.0, 'EXPOSURE', n)

total_infected = 0
cases = set()

tau = 0.05
gamma = 1.0 / 6.0
sigma = 1.0 / 8.5

while queue and total_infected < 400:
    t, _, event_type, target = heapq.heappop(queue)
    
    if event_type == 'EXPOSURE':
        if status[target] != 'S': continue
        status[target] = 'E'
        cases.add(target)
        total_infected += 1
        onset_time = t + np.random.exponential(1.0 / sigma)
        add_event(onset_time, 'ONSET', target)
        
    elif event_type == 'EXPOSURE_CANDIDATE':
        if status[target] != 'S': continue
        status[target] = 'E'
        cases.add(target)
        total_infected += 1
        onset_time = t + np.random.exponential(1.0 / sigma)
        add_event(onset_time, 'ONSET', target)
        
    elif event_type == 'ONSET':
        if status[target] == 'E':
            status[target] = 'I'
            rec_t = t + np.random.exponential(1.0 / gamma)
            add_event(rec_t, 'RECOVERY', target)
            
            for neighbor in G.neighbors(target):
                if status[neighbor] == 'S':
                    inf_time = t + np.random.exponential(1.0 / tau)
                    if inf_time < rec_t:
                        add_event(inf_time, 'EXPOSURE_CANDIDATE', neighbor)
                        
    elif event_type == 'RECOVERY':
        if status[target] == 'I':
            status[target] = 'R'

print(f"Simulation stopped at {total_infected} cases. Computing spring layout...")
pos = nx.spring_layout(G, k=0.12, iterations=25, seed=42)

print("Plotting TRUE network on white background...")
fig, ax = plt.subplots(figsize=(16, 16), facecolor='white', dpi=300)
ax.set_facecolor('white')

cases_list = list(cases)
healthy_list = [n for n in G.nodes() if n not in cases]

# Draw edges (very faint)
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
filename = f"figures/real_network_visualization_{timestamp}.png"

plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
print(f"Saved visualization to: {filename}")
