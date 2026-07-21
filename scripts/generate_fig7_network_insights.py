import numpy as np
import matplotlib.pyplot as plt
import datetime
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
import networkx as nx

print("Generating Network Insights Figure (Transmission Trees)...")

N = 5000
G = generate_network(N)
tau = calibrate_tau(G, 1.6, 1.0/6.0)

scenarios = [
    {"name": "Base Case (No Intervention)", "radius": 1, "eff": 0.0, "risk": 1.0, "rr": 0.0},
    {"name": "Ring 1 (Household Only)", "radius": 1, "eff": 0.7, "risk": 1.0, "rr": 1.0},
    {"name": "Ring 2 (Household + Community)", "radius": 2, "eff": 0.7, "risk": 1.0, "rr": 1.0}
]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)

for i, sc in enumerate(scenarios):
    print(f"Processing {sc['name']}...")
    
    rep_edges = []
    # Find a representative outbreak (stop at 100 cases max)
    for _ in range(100):
        edges = simulate_ring_vaccination(
            G, baseline_tau=tau, ring_radius=sc['radius'], 
            efficacy=sc['eff'], risk_compensation_multiplier=sc['risk'],
            reporting_rate=sc['rr'], return_transmission_network=True,
            initial_infected=3, max_cases=100
        )
        rep_edges = edges
        # We want a tree that isn't completely dead but isn't a massive hairball. 40-100 is good for Base/Ring1.
        # For Ring 2, it might die faster, so we accept anything if it's the last try
        if 40 <= len(edges) <= 100:
            break
            
    T = nx.DiGraph()
    for u, v, w, t in rep_edges:
        T.add_edge(u, v, weight=w)
        
    ax_tree = axes[i]
    ax_tree.set_title(sc['name'], fontsize=18, fontweight='bold', pad=15)
    
    if len(T.nodes()) > 0:
        try:
            pos = nx.nx_agraph.graphviz_layout(T, prog="dot")
        except:
            try:
                pos = nx.kamada_kawai_layout(T)
            except:
                pos = nx.spring_layout(T, seed=42)
            
        edge_colors = ['#0044cc' if T[u][v]['weight'] == 3.0 else '#cc0000' for u, v in T.edges()]
        # Draw Nodes
        nx.draw_networkx_nodes(T, pos, ax=ax_tree, node_color='black', node_size=30, alpha=0.8, edgecolors='white')
        # Draw Edges
        nx.draw_networkx_edges(T, pos, ax=ax_tree, edge_color=edge_colors, width=2.0, alpha=0.8, arrows=True, arrowsize=12)
    else:
        ax_tree.text(0.5, 0.5, "Outbreak Extinguished Immediately", ha='center', va='center', fontsize=14)
        
    ax_tree.axis('off')

plt.suptitle("Transmission Trees: Dense Clusters vs Long Community Escapes", fontsize=24, fontweight='bold', y=1.05)
plt.tight_layout()

# Legend
import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [], color='#0044cc', label='Household Transmission (Close)', linewidth=3)
red_line = mlines.Line2D([], [], color='#cc0000', label='Community Jump (General)', linewidth=3)
fig.legend(handles=[blue_line, red_line], loc='lower center', ncol=2, fontsize=16, bbox_to_anchor=(0.5, -0.05))

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/fig7_network_insights_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved {filename}")
