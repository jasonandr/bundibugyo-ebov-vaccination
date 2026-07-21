import numpy as np
import matplotlib.pyplot as plt
import datetime
from ebola_stochastic_ring import generate_network, calibrate_tau, simulate_ring_vaccination
import networkx as nx

print("Generating 100-Layer Dense Adjacency Fingerprints...")

N = 5000
G = generate_network(N)
tau = calibrate_tau(G, 1.6, 1.0/6.0)
hh_dict = nx.get_node_attributes(G, 'household')

scenarios = [
    {"name": "Base Case (No Intervention)", "radius": 1, "eff": 0.0, "risk": 1.0, "rr": 0.0},
    {"name": "Ring Vaccination (Radius 1)", "radius": 1, "eff": 0.7, "risk": 1.0, "rr": 1.0},
    {"name": "General / Mass Vaccination", "radius": 10, "eff": 0.7, "risk": 1.0, "rr": 1.0},
    {"name": "Risk Compensation Paradox", "radius": 1, "eff": 0.3, "risk": 3.0, "rr": 1.0}
]

max_matrix_size = 500
num_layers = 100

fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=300)
axes = axes.flatten()

for i, sc in enumerate(scenarios):
    print(f"Running Scenario: {sc['name']} ({num_layers} layers)...")
    
    hh_x, hh_y = [], []
    comm_x, comm_y = [], []
    
    runs_completed = 0
    while runs_completed < num_layers:
        edges = simulate_ring_vaccination(
            G, baseline_tau=tau, 
            ring_radius=sc['radius'], 
            efficacy=sc['eff'], 
            risk_compensation_multiplier=sc['risk'],
            reporting_rate=sc['rr'],
            return_transmission_network=True,
            initial_infected=5,
            max_cases=max_matrix_size
        )
        
        # Skip total duds to focus on the structure of outbreaks that actually take off
        if len(edges) < 15:
            continue
            
        runs_completed += 1
        
        infected_nodes = set()
        infection_times = {}
        for u, v, w, t in edges:
            infected_nodes.add(u)
            infected_nodes.add(v)
            if v not in infection_times:
                infection_times[v] = t
        
        for n in infected_nodes:
            if n not in infection_times:
                infection_times[n] = 0.0

        sorted_nodes = sorted(list(infected_nodes), key=lambda x: (hh_dict.get(x, -1), infection_times[x]))
        node_to_idx = {n: j for j, n in enumerate(sorted_nodes)}
        
        for u, v, w, t in edges:
            if w == 3.0:
                hh_x.append(node_to_idx[u])
                hh_y.append(node_to_idx[v])
            else:
                comm_x.append(node_to_idx[u])
                comm_y.append(node_to_idx[v])
                
    ax = axes[i]
    ax.set_facecolor('#ffffff') # White background for legibility
    
    # Use edgecolors='none' so tiny scatter points don't just draw black borders
    ax.scatter(hh_x, hh_y, c='#0044cc', s=4, alpha=0.08, edgecolors='none', label="Household Spread (Close)" if i==0 else "")
    ax.scatter(comm_x, comm_y, c='#cc0000', s=4, alpha=0.08, edgecolors='none', label="Community Jump (General)" if i==0 else "")
    
    # Fixed axes dimensions for side-by-side scale comparison
    ax.set_xlim(-5, max_matrix_size + 5)
    ax.set_ylim(-5, max_matrix_size + 5)
    ax.invert_yaxis()
    
    ax.set_title(f"{sc['name']}", fontsize=18, fontweight='bold', pad=15, color='black')
    ax.set_xlabel("Infectee Index (Sorted by Household)", fontsize=12)
    ax.set_ylabel("Infector Index (Sorted by Household)", fontsize=12)
    
    if i == 0:
        leg = ax.legend(fontsize=14, loc='upper right', facecolor='white', framealpha=0.9)
        for lh in leg.legend_handles: 
            lh.set_alpha(1)

plt.suptitle(f"Transmission Structure Fingerprints ({num_layers} Overlaid Simulations)", fontsize=24, fontweight='bold', y=1.02)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/fig7_adjacency_layered_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved {filename}")
