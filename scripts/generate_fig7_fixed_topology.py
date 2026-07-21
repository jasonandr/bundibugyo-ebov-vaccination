import numpy as np
import matplotlib.pyplot as plt
import datetime
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
import networkx as nx

print("Generating Fixed-Topology Adjacency Fingerprints...")

# 1. FIXED GLOBAL TOPOLOGY
N = 500
G = generate_network(N)
tau = calibrate_tau(G, 1.6, 1.0/6.0)
hh_dict = nx.get_node_attributes(G, 'household')

# Sort all global nodes permanently by household
sorted_global_nodes = sorted(list(G.nodes()), key=lambda x: hh_dict.get(x, -1))
global_node_to_idx = {n: i for i, n in enumerate(sorted_global_nodes)}

scenarios = [
    {"name": "Base Case (No Intervention)", "radius": 1, "eff": 0.0, "risk": 1.0, "rr": 0.0},
    {"name": "Ring Vaccination (Radius 1)", "radius": 1, "eff": 0.7, "risk": 1.0, "rr": 1.0},
    {"name": "Ring Vaccination (Radius 2)", "radius": 2, "eff": 0.7, "risk": 1.0, "rr": 1.0},
    {"name": "Risk Compensation Paradox", "radius": 1, "eff": 0.3, "risk": 3.0, "rr": 1.0}
]

num_layers = 100

fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=300)
axes = axes.flatten()

for i, sc in enumerate(scenarios):
    print(f"Running Scenario: {sc['name']}...")
    
    hh_x, hh_y = [], []
    comm_x, comm_y = [], []
    
    for r in range(num_layers):
        # We don't skip duds! We plot every single run to get true expected network density
        edges = simulate_ring_vaccination(
            G, baseline_tau=tau, 
            ring_radius=sc['radius'], 
            efficacy=sc['eff'], 
            risk_compensation_multiplier=sc['risk'],
            reporting_rate=sc['rr'],
            return_transmission_network=True,
            initial_infected=5,
            max_cases=N
        )
        
        for u, v, w, t in edges:
            if w == 3.0:
                hh_x.append(global_node_to_idx[u])
                hh_y.append(global_node_to_idx[v])
            else:
                comm_x.append(global_node_to_idx[u])
                comm_y.append(global_node_to_idx[v])
                
    ax = axes[i]
    ax.set_facecolor('#ffffff') # White background
    
    # Overlay all 100 runs. Use size=3, alpha=0.1. No black edgecolors.
    ax.scatter(hh_x, hh_y, c='#0044cc', s=3, alpha=0.1, edgecolors='none', label="Household Spread (Close)" if i==0 else "")
    ax.scatter(comm_x, comm_y, c='#cc0000', s=3, alpha=0.1, edgecolors='none', label="Community Spread (General)" if i==0 else "")
    
    ax.set_xlim(-5, N + 5)
    ax.set_ylim(-5, N + 5)
    ax.invert_yaxis()
    
    ax.set_title(f"{sc['name']}", fontsize=18, fontweight='bold', pad=15, color='black')
    ax.set_xlabel("Global Infectee Index", fontsize=12)
    ax.set_ylabel("Global Infector Index", fontsize=12)
    
    if i == 0:
        leg = ax.legend(fontsize=14, loc='upper right', facecolor='white', framealpha=0.9)
        for lh in leg.legend_handles: 
            lh.set_alpha(1)

plt.suptitle(f"Expected Transmission Networks ({num_layers} Stochastic Runs on Fixed N=500 Community)", fontsize=22, fontweight='bold', y=1.02)
plt.tight_layout()

# Cache busting dynamic filename
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/fig7_fixed_adjacency_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved {filename}")
