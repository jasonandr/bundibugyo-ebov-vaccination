import networkx as nx
import matplotlib.pyplot as plt
import datetime
import os

print("Generating Idea 1: Network Paradox Diagram...")
G = nx.Graph()

# Create a central "Index Cluster"
G.add_edges_from([(0, i) for i in range(1, 6)])

# Create the "Ring" connecting to the community
for i in range(1, 6):
    G.add_edges_from([(i, 5 + i * 2), (i, 6 + i * 2)])
    
# Add some cross-community links
G.add_edge(7, 8)
G.add_edge(9, 10)
G.add_edge(11, 12)
G.add_edge(13, 14)
G.add_edge(15, 16)

pos = nx.spring_layout(G, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

for i, ax in enumerate(axes):
    ax.set_facecolor('white')
    
    # Base nodes
    index_nodes = [0]
    ring_nodes = list(range(1, 6))
    community_nodes = list(range(7, 17))
    
    if i == 0:
        ax.set_title("Scenario A: Successful Ring Vaccination", fontsize=16, fontweight='bold')
        # All ring nodes are green
        nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, node_color='lightgray', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=ring_nodes, node_color='#2ca02c', node_size=400, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=index_nodes, node_color='#d62728', node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='#dddddd', width=2, ax=ax)
        
        # Red edges stopped at green nodes
        stopped_edges = [(0, r) for r in ring_nodes]
        nx.draw_networkx_edges(G, pos, edgelist=stopped_edges, edge_color='#d62728', width=3, ax=ax)
        
    else:
        ax.set_title("Scenario B: Risk Compensation Paradox", fontsize=16, fontweight='bold')
        # One ring node gets infected due to paradox
        failed_ring_node = 2
        paradox_infected = [failed_ring_node]
        safe_ring_nodes = [n for n in ring_nodes if n != failed_ring_node]
        
        # Community nodes infected
        new_cluster = [9, 10]
        safe_community = [n for n in community_nodes if n not in new_cluster]
        
        nx.draw_networkx_nodes(G, pos, nodelist=safe_community, node_color='lightgray', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=safe_ring_nodes, node_color='#2ca02c', node_size=400, ax=ax)
        
        # Draw failed node
        nx.draw_networkx_nodes(G, pos, nodelist=paradox_infected, node_color='#ff7f0e', node_size=450, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=new_cluster, node_color='#d62728', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=index_nodes, node_color='#d62728', node_size=500, ax=ax)
        
        nx.draw_networkx_edges(G, pos, edge_color='#dddddd', width=2, ax=ax)
        
        # Red edges jumping
        jump_edges = [(0, failed_ring_node), (failed_ring_node, 9), (failed_ring_node, 10), (9, 10)]
        nx.draw_networkx_edges(G, pos, edgelist=jump_edges, edge_color='#d62728', width=3, ax=ax)
        
    ax.axis('off')

plt.tight_layout()
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea1_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
print(f"Saved {filename}")
