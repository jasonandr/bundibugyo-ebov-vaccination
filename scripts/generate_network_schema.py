import networkx as nx
import matplotlib.pyplot as plt
import os
import time

def generate_network_schema():
    # Create a small network with 3 household cliques
    G = nx.Graph()
    
    # Household 1 (size 5)
    hh1 = [1, 2, 3, 4, 5]
    for u in hh1:
        for v in hh1:
            if u != v:
                G.add_edge(u, v, layer='household')
                
    # Household 2 (size 4)
    hh2 = [6, 7, 8, 9]
    for u in hh2:
        for v in hh2:
            if u != v:
                G.add_edge(u, v, layer='household')
                
    # Household 3 (size 5)
    hh3 = [10, 11, 12, 13, 14]
    for u in hh3:
        for v in hh3:
            if u != v:
                G.add_edge(u, v, layer='household')
                
    # Add community spread (heavy tail/superspreading)
    # Node 1 is a superspreader connecting to other households
    G.add_edge(1, 6, layer='community')
    G.add_edge(1, 10, layer='community')
    G.add_edge(7, 14, layer='community')
    
    pos = {}
    # Position households in clusters
    pos.update(nx.spring_layout(G.subgraph(hh1), center=(-2, 1), scale=0.5))
    pos.update(nx.spring_layout(G.subgraph(hh2), center=(2, 1), scale=0.5))
    pos.update(nx.spring_layout(G.subgraph(hh3), center=(0, -2), scale=0.5))
    
    household_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['layer'] == 'household']
    community_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['layer'] == 'community']
    
    plt.figure(figsize=(8, 6))
    
    # Draw households
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=household_edges, width=2.0, alpha=0.8, edge_color='gray')
    
    # Draw community links (dashed, red)
    nx.draw_networkx_edges(G, pos, edgelist=community_edges, width=2.5, alpha=0.9, edge_color='red', style='dashed')
    
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title("Model Schema: Two-Layer Household-Community Network", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    timestamp = int(time.time())
    img_path = os.path.join("figures", f"schema_network_{timestamp}.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    print(f"Saved Network Schema to {img_path}")

if __name__ == "__main__":
    generate_network_schema()
