import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime
import itertools
import heapq

print("Generating 10,000 node TWO-LAYER network with moderate visual clustering weights...")
N = 10000
G = nx.Graph()
G.add_nodes_from(range(N))

# --- Layer 1: Household Cliques ---
nodes = list(G.nodes())
np.random.shuffle(nodes)
idx = 0
while idx < N:
    hh_size = np.random.poisson(4) + 1
    if idx + hh_size > N:
        hh_size = N - idx
    hh_nodes = nodes[idx:idx+hh_size]
    
    for u, v in itertools.combinations(hh_nodes, 2):
        G.add_edge(u, v, weight=3.0)
    idx += hh_size

# --- Layer 2: Community Spread ---
mean_k = 5.0
var_k = 25.0
p_nb = mean_k / var_k
r_nb = mean_k**2 / (var_k - mean_k)

ks = np.random.negative_binomial(r_nb, p_nb, N)
if sum(ks) % 2 != 0:
    ks[0] += 1
    
comm_G = nx.configuration_model(ks)
comm_G = nx.Graph(comm_G)
comm_G.remove_edges_from(nx.selfloop_edges(comm_G))

for u, v in comm_G.edges():
    if not G.has_edge(u, v):
        G.add_edge(u, v, weight=1.0)
        
print("Computing spring layout (weighted)...")
pos = nx.spring_layout(G, k=0.15, iterations=30, weight='weight', seed=42)

print("Running SEIR simulation to capture snapshots...")

def run_simulation():
    attempts = 0
    while True:
        attempts += 1
        status = {n: 'S' for n in G.nodes()}
        queue = []
        counter = itertools.count()

        def add_event(t, event_type, target):
            heapq.heappush(queue, (t, next(counter), event_type, target))

        # Start randomly as requested
        initial_node = np.random.choice(G.nodes())
        add_event(0.0, 'EXPOSURE', initial_node)

        total_onsets = 0
        
        tau = 0.05
        gamma = 1.0 / 6.0
        sigma = 1.0 / 8.5

        snapshots = []
        target_milestones = [1, 10, 50, 400]
        current_milestone_idx = 0

        while queue and current_milestone_idx < len(target_milestones):
            t, _, event_type, target = heapq.heappop(queue)
            
            if event_type == 'EXPOSURE' or event_type == 'EXPOSURE_CANDIDATE':
                if status[target] != 'S': continue
                status[target] = 'E'
                onset_time = t + np.random.exponential(1.0 / sigma)
                add_event(onset_time, 'ONSET', target)
                
            elif event_type == 'ONSET':
                if status[target] == 'E':
                    status[target] = 'I'
                    total_onsets += 1
                    
                    if total_onsets == target_milestones[current_milestone_idx]:
                        # Capture ONLY current active I cases
                        current_i_cases = {n for n, s in status.items() if s == 'I'}
                        snapshots.append(set(current_i_cases))
                        current_milestone_idx += 1
                        
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
                    
        if len(snapshots) == 4:
            print(f"Successful outbreak reached {target_milestones[-1]} onsets after {attempts} attempts.")
            return snapshots

snapshots = run_simulation()

styles = [
    {"name": "VarA_Purple_Gray", "edge_c": "#CCCCCC", "node_c": "purple", "node_a": 0.3, "case_s": 25, "case_c": "red"},
    {"name": "VarB_Indigo_LightGray", "edge_c": "#EEEEEE", "node_c": "indigo", "node_a": 0.2, "case_s": 35, "case_c": "#FF1111"},
    {"name": "VarC_BlueViolet_Gray", "edge_c": "#E5E5E5", "node_c": "#8A2BE2", "node_a": 0.25, "case_s": 30, "case_c": "#FF3333"}
]

x_vals = [p[0] for p in pos.values()]
y_vals = [p[1] for p in pos.values()]
xlims = (np.percentile(x_vals, 2), np.percentile(x_vals, 98))
ylims = (np.percentile(y_vals, 2), np.percentile(y_vals, 98))
titles = ["Index Case (1)", "Early Spread (10 Cumulative Onsets)", "Emerging Cluster (50 Cumulative)", "Outbreak Peak (400 Cumulative)"]

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

for style in styles:
    print(f"Plotting variation: {style['name']}...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), facecolor='white', dpi=200)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.set_facecolor('white')
        snap_cases = snapshots[i]
        cases_list = list(snap_cases)
        healthy_list = [n for n in G.nodes() if n not in snap_cases]
        
        nx.draw_networkx_edges(G, pos, alpha=0.04, edge_color=style["edge_c"], ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=healthy_list, node_size=2, node_color=style["node_c"], alpha=style["node_a"], ax=ax)
        
        if cases_list:
            nx.draw_networkx_nodes(G, pos, nodelist=cases_list, node_size=style["case_s"], node_color=style["case_c"], alpha=1.0, ax=ax)
            
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(titles[i], fontsize=20, fontweight='bold')
        ax.axis('off')

    filename = f"figures/network_progression_{style['name']}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to: {filename}")
