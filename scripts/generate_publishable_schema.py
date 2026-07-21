import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import os
import time

def draw_network_schema(ax):
    G = nx.Graph()
    # Create 4 households
    hhs = [[1,2,3,4], [5,6,7,8,9], [10,11,12], [13,14,15,16,17]]
    for hh in hhs:
        for u in hh:
            for v in hh:
                if u != v:
                    G.add_edge(u, v, layer='household')
    
    # Add community spread
    G.add_edge(1, 5, layer='community')
    G.add_edge(2, 10, layer='community')
    G.add_edge(8, 14, layer='community')
    G.add_edge(12, 17, layer='community')
    
    pos = {}
    pos.update(nx.spring_layout(G.subgraph(hhs[0]), center=(-2, 2), scale=0.6))
    pos.update(nx.spring_layout(G.subgraph(hhs[1]), center=(2, 2), scale=0.7))
    pos.update(nx.spring_layout(G.subgraph(hhs[2]), center=(-2, -2), scale=0.5))
    pos.update(nx.spring_layout(G.subgraph(hhs[3]), center=(2, -2), scale=0.7))
    
    # Tweak position to spread them out a bit more cleanly
    # (Just let spring layout handle the micro layout, center handles macro)
    
    household_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['layer'] == 'household']
    community_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['layer'] == 'community']
    
    # Draw households
    nx.draw_networkx_nodes(G, pos, node_color='#4A90E2', node_size=300, edgecolors='white', linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=household_edges, width=1.5, alpha=0.6, edge_color='#7F8C8D', ax=ax)
    
    # Draw community links (dashed, red)
    nx.draw_networkx_edges(G, pos, edgelist=community_edges, width=2.0, alpha=0.9, edge_color='#E74C3C', style='dashed', ax=ax)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#7F8C8D', lw=1.5, label='Household / Close Contact'),
        Line2D([0], [0], color='#E74C3C', lw=2.0, linestyle='dashed', label='Community / Superspreading')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)
    
    ax.set_title("A. Two-Layer Transmission Network", loc='left', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

def draw_seir_timeline(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_title("B. Natural History & Intervention Timeline", loc='left', fontsize=14, fontweight='bold', pad=15)
    
    # Draw SEIR Blocks
    y_seir = 70
    boxes = [
        ("Susceptible\n(S)", 10, y_seir, '#4A90E2'),
        ("Exposed\n(E)", 35, y_seir, '#F39C12'),
        ("Infectious\n(I)", 60, y_seir, '#E74C3C'),
        ("Removed\n(R)", 85, y_seir, '#7F8C8D')
    ]
    
    for text, x, y, color in boxes:
        box = patches.FancyBboxPatch((x-6, y-6), 12, 12, boxstyle="round,pad=0.5", ec='white', fc=color, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
    # SEIR Arrows
    ax.annotate("", xy=(29, y_seir), xytext=(22, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
    ax.text(25.5, y_seir+2, r'$\lambda$', ha='center', va='bottom', fontsize=12)
    
    ax.annotate("", xy=(54, y_seir), xytext=(47, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
    ax.text(50.5, y_seir+2, r'$\sigma$', ha='center', va='bottom', fontsize=12)
    
    ax.annotate("", xy=(79, y_seir), xytext=(72, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
    ax.text(75.5, y_seir+2, r'$\gamma$', ha='center', va='bottom', fontsize=12)
    
    # Draw Timeline
    y_time = 30
    ax.plot([10, 90], [y_time, y_time], color='#BDC3C7', lw=4, zorder=1)
    
    events = [
        ("Exposure", 10, "Transmission"),
        ("Onset", 35, "Incubation Period"),
        ("Detection", 50, "Reporting Rate"),
        ("Vaccination", 70, "Tracing Delay"),
        ("Immunity", 90, "Immune Delay")
    ]
    
    # Tick marks
    for label, x, desc in events:
        ax.plot([x, x], [y_time-3, y_time+3], color='#2C3E50', lw=2, zorder=2)
        ax.text(x, y_time-6, label, ha='center', va='top', fontweight='bold', fontsize=10, color='#2C3E50')
        
    # Interval brackets/text
    ax.annotate("", xy=(10, y_time+5), xytext=(35, y_time+5), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#7F8C8D'))
    ax.text(22.5, y_time+8, "Incubation Period", ha='center', va='bottom', fontsize=9, color='#7F8C8D')
    
    ax.annotate("", xy=(35, y_time+5), xytext=(50, y_time+5), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#7F8C8D'))
    ax.text(42.5, y_time+8, "Detection Delay", ha='center', va='bottom', fontsize=9, color='#7F8C8D')
    
    ax.annotate("", xy=(50, y_time+5), xytext=(70, y_time+5), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#F39C12'))
    ax.text(60, y_time+8, "Tracing Delay", ha='center', va='bottom', fontsize=9, color='#F39C12', fontweight='bold')
    
    ax.annotate("", xy=(70, y_time+5), xytext=(90, y_time+5), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#27AE60'))
    ax.text(80, y_time+8, "Immune Delay", ha='center', va='bottom', fontsize=9, color='#27AE60', fontweight='bold')
    
    # Connecting dashed line from SEIR to Timeline
    ax.plot([35, 35], [y_seir-6, y_time+12], ls=':', color='#BDC3C7', lw=1.5)
    ax.plot([10, 10], [y_seir-6, y_time+12], ls=':', color='#BDC3C7', lw=1.5)

def create_publishable_schema():
    # Set global font to sans-serif (Arial-like)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    draw_network_schema(ax1)
    draw_seir_timeline(ax2)
    
    plt.tight_layout(pad=3.0)
    
    timestamp = int(time.time())
    out_dir = "figures"
    img_path = os.path.join(out_dir, f"publishable_schema_{timestamp}.png")
    
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved High-Res Publishable Schema to {img_path}")

if __name__ == "__main__":
    create_publishable_schema()
