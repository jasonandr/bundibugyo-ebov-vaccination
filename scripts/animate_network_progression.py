import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import datetime
import itertools
import heapq

print("Generating 10,000 node network for animation...")
N = 10000
G = nx.Graph()
G.add_nodes_from(range(N))

nodes = list(G.nodes())
np.random.shuffle(nodes)
idx = 0
while idx < N:
    hh_size = np.random.poisson(4) + 1
    if idx + hh_size > N: hh_size = N - idx
    hh_nodes = nodes[idx:idx+hh_size]
    for u, v in itertools.combinations(hh_nodes, 2):
        G.add_edge(u, v, weight=3.0)
    idx += hh_size

mean_k = 5.0
var_k = 25.0
p_nb = mean_k / var_k
r_nb = mean_k**2 / (var_k - mean_k)
ks = np.random.negative_binomial(r_nb, p_nb, N)
if sum(ks) % 2 != 0: ks[0] += 1
comm_G = nx.Graph(nx.configuration_model(ks))
comm_G.remove_edges_from(nx.selfloop_edges(comm_G))
for u, v in comm_G.edges():
    if not G.has_edge(u, v):
        G.add_edge(u, v, weight=1.0)

print("Computing spring layout...")
pos = nx.spring_layout(G, k=0.15, iterations=30, weight='weight', seed=42)

print("Running simulation for animation frames...")
def run_simulation_frames():
    attempts = 0
    while True:
        attempts += 1
        status = {n: 'S' for n in G.nodes()}
        queue = []
        counter = itertools.count()

        def add_event(t, event_type, target):
            heapq.heappush(queue, (t, next(counter), event_type, target))

        initial_node = np.random.choice(G.nodes())
        add_event(0.0, 'EXPOSURE', initial_node)

        total_onsets = 0
        tau = 0.05
        gamma = 1.0 / 6.0
        sigma = 1.0 / 8.5

        frames = [] # list of (day, active_cases_set)
        current_day = 0.0

        while queue and total_onsets < 400:
            t, _, event_type, target = heapq.heappop(queue)
            
            while t >= current_day:
                current_i = {n for n, s in status.items() if s == 'I'}
                frames.append((current_day, current_i))
                current_day += 1.0
                
            if event_type == 'EXPOSURE' or event_type == 'EXPOSURE_CANDIDATE':
                if status[target] != 'S': continue
                status[target] = 'E'
                onset_time = t + np.random.exponential(1.0 / sigma)
                add_event(onset_time, 'ONSET', target)
                
            elif event_type == 'ONSET':
                if status[target] == 'E':
                    status[target] = 'I'
                    total_onsets += 1
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
                    
        if total_onsets >= 400:
            # Capture the remaining frames up to current t
            while current_day <= t:
                current_i = {n for n, s in status.items() if s == 'I'}
                frames.append((current_day, current_i))
                current_day += 1.0
            print(f"Simulation success on attempt {attempts}.")
            return frames

frames = run_simulation_frames()

print(f"Generating animation with {len(frames)} frames...")
fig, ax = plt.subplots(figsize=(16, 16), facecolor='white', dpi=120)
ax.set_facecolor('white')

# Var C aesthetics
edge_color = "#E5E5E5"
node_color = "#8A2BE2"
node_alpha = 0.25
case_size = 40
case_color = "#FF3333"

x_vals = [p[0] for p in pos.values()]
y_vals = [p[1] for p in pos.values()]
ax.set_xlim(np.percentile(x_vals, 2), np.percentile(x_vals, 98))
ax.set_ylim(np.percentile(y_vals, 2), np.percentile(y_vals, 98))
ax.axis('off')

# Background
nx.draw_networkx_edges(G, pos, alpha=0.04, edge_color=edge_color, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=2, node_color=node_color, alpha=node_alpha, ax=ax)

case_scatter = ax.scatter([], [], s=case_size, c=case_color, zorder=5)
title_text = ax.set_title("Day: 0 | Active Cases: 0", fontsize=24, fontweight='bold')

def init():
    case_scatter.set_offsets(np.empty((0, 2)))
    title_text.set_text("Day: 0 | Active Cases: 0")
    return case_scatter, title_text

def update(frame_data):
    day, active_nodes = frame_data
    if active_nodes:
        x = [pos[n][0] for n in active_nodes]
        y = [pos[n][1] for n in active_nodes]
        case_scatter.set_offsets(np.c_[x, y])
    else:
        case_scatter.set_offsets(np.empty((0, 2)))
    
    title_text.set_text(f"Day: {int(day)} | Active Cases: {len(active_nodes)}")
    return case_scatter, title_text

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=250)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/outbreak_animation_VarC_{timestamp}.mp4"

writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='EbolaModel'), bitrate=1800)
ani.save(filename, writer=writer)
plt.close(fig)
print(f"Animation saved to {filename}")
