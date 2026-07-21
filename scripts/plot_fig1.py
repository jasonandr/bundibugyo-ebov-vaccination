import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import time
import os
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from paths import figure_path, result_path
from current_outbreak_data import cumulative_confirmed_cases, cumulative_confirmed_deaths
from ebola_stochastic_ring import generate_network

def main():
    print("Loading 2026 parameters and empirical data...")
    with open(result_path("fitted_parameters.json"), "r") as f:
        params = json.load(f)
        
    rt_array = params.get("Rt_array", [])
    
    # Process data
    cases_df = cumulative_confirmed_cases()
    deaths_df = cumulative_confirmed_deaths()
    
    cases_df = cases_df.sort_values('Date')
    deaths_df = deaths_df.sort_values('Date')
    
    # Enforce monotonicity
    cases_raw = cases_df['Cases'].values.copy()
    for i in range(len(cases_raw)-2, -1, -1):
        if cases_raw[i] > cases_raw[i+1]:
            cases_raw[i] = cases_raw[i+1]
    cases_df['Cases'] = cases_raw
    
    date_range = pd.date_range(start=cases_df['Date'].min(), end=cases_df['Date'].max(), freq='D')
    df_daily = pd.DataFrame({'Date': date_range})
    df_daily = pd.merge(df_daily, cases_df, on='Date', how='left')
    df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')
    cases_cum = df_daily['Cases'].values
    cases_inc = np.diff(cases_cum, prepend=cases_cum[0])
    
    deaths_raw = deaths_df['Deaths'].values.copy()
    for i in range(len(deaths_raw)-2, -1, -1):
        if deaths_raw[i] > deaths_raw[i+1]:
            deaths_raw[i] = deaths_raw[i+1]
    deaths_df['Deaths'] = deaths_raw
    df_daily_deaths = pd.merge(pd.DataFrame({'Date': date_range}), deaths_df, on='Date', how='left')
    df_daily_deaths['Deaths'] = df_daily_deaths['Deaths'].interpolate(method='linear').fillna(0)
    deaths_cum = df_daily_deaths['Deaths'].values
    deaths_inc = np.diff(deaths_cum, prepend=deaths_cum[0])
    
    fig = plt.figure(figsize=(24, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # ---------------------------------------------------------
    # Panel A: SEIR Schema
    # ---------------------------------------------------------
    ax1 = plt.subplot(gs[0])
    ax1.axis('off')
    ax1.set_title("A", loc="left", fontsize=24, fontweight="bold")
    
    # Draw boxes
    boxes = [
        ("Susceptible\n(S)", "#4A90E2", 0.1),
        ("Exposed\n(E)", "#F39C12", 0.35),
        ("Infectious\n(I)", "#E74C3C", 0.6),
        ("Removed\n(R)", "#7F8C8D", 0.85)
    ]
    box_w, box_h = 0.18, 0.15
    for text, color, x in boxes:
        rect = plt.Rectangle((x - box_w/2, 0.6), box_w, box_h, color=color, zorder=2)
        ax1.add_patch(rect)
        ax1.text(x, 0.6 + box_h/2, text, color="white", ha="center", va="center", fontweight="bold", fontsize=12, zorder=3)
        
    # Draw arrows
    for i in range(3):
        x_start = boxes[i][2] + box_w/2 + 0.01
        x_end = boxes[i+1][2] - box_w/2 - 0.01
        ax1.annotate("", xy=(x_end, 0.6 + box_h/2), xytext=(x_start, 0.6 + box_h/2),
                     arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2))
        
    ax1.text((boxes[0][2] + boxes[1][2])/2, 0.6 + box_h/2 + 0.03, r"$\lambda$", ha="center", fontsize=14)
    ax1.text((boxes[1][2] + boxes[2][2])/2, 0.6 + box_h/2 + 0.03, r"$\sigma$", ha="center", fontsize=14)
    ax1.text((boxes[2][2] + boxes[3][2])/2, 0.6 + box_h/2 + 0.03, r"$\gamma$", ha="center", fontsize=14)

    # Timeline points
    y_timeline = 0.3
    events = [
        (0.1, "Exposure"),
        (0.35, "Onset"),
        (0.5, "Detection"),
        (0.7, "Vaccinate"),
        (0.9, "Immunity")
    ]
    ax1.plot([0.1, 0.9], [y_timeline, y_timeline], color="#2C3E50", lw=2)
    for x, label in events:
        ax1.plot([x, x], [y_timeline - 0.02, y_timeline + 0.02], color="#2C3E50", lw=2)
        ax1.text(x, y_timeline - 0.04, label, ha="center", va="top", fontweight="bold", fontsize=12, color="#2C3E50")
        
    # Vertical dotted lines mapping boxes to timeline
    ax1.plot([0.1, 0.1], [y_timeline, 0.6], color="#BDC3C7", ls=":", lw=2)
    ax1.plot([0.35, 0.35], [y_timeline, 0.6], color="#BDC3C7", ls=":", lw=2)
    
    # Intervals
    ax1.annotate("", xy=(0.35, y_timeline + 0.04), xytext=(0.1, y_timeline + 0.04), arrowprops=dict(arrowstyle="<->", color="#7F8C8D"))
    ax1.text(0.225, y_timeline + 0.06, "Incubation\nPeriod", ha="center", va="bottom", fontsize=10, color="#7F8C8D")
    
    ax1.annotate("", xy=(0.5, y_timeline + 0.04), xytext=(0.35, y_timeline + 0.04), arrowprops=dict(arrowstyle="<->", color="#7F8C8D"))
    ax1.text(0.425, y_timeline + 0.06, "Reporting\nDelay", ha="center", va="bottom", fontsize=10, color="#7F8C8D")
    
    ax1.annotate("", xy=(0.7, y_timeline + 0.04), xytext=(0.5, y_timeline + 0.04), arrowprops=dict(arrowstyle="<->", color="#F39C12"))
    ax1.text(0.6, y_timeline + 0.06, "Tracing\nDelay", ha="center", va="bottom", fontsize=10, color="#F39C12", fontweight="bold")
    
    ax1.annotate("", xy=(0.9, y_timeline + 0.04), xytext=(0.7, y_timeline + 0.04), arrowprops=dict(arrowstyle="<->", color="#27AE60"))
    ax1.text(0.8, y_timeline + 0.06, "Immune\nDelay", ha="center", va="bottom", fontsize=10, color="#27AE60", fontweight="bold")
    
    # Indicate who events pertain to (User's request)
    ax1.text(0.3, y_timeline - 0.12, "Index Cases", ha="center", fontsize=12, fontweight="bold", color="#E74C3C")
    ax1.plot([0.1, 0.5], [y_timeline - 0.10, y_timeline - 0.10], color="#E74C3C", lw=2)
    
    ax1.text(0.8, y_timeline - 0.12, "Contacts", ha="center", fontsize=12, fontweight="bold", color="#F39C12")
    ax1.plot([0.7, 0.9], [y_timeline - 0.10, y_timeline - 0.10], color="#F39C12", lw=2)

    # ---------------------------------------------------------
    # Panel B: Ring Schema
    # ---------------------------------------------------------
    ax2 = plt.subplot(gs[1])
    ax2.set_title("B", loc="left", fontsize=24, fontweight="bold")
    ax2.axis('off')
    G = generate_network(N=300, household_mean=3.0, community_mean=4.0, community_variance=10.0)
    degrees = dict(G.degree())
    index_case = max(degrees, key=degrees.get)
    
    rad1 = set(G.neighbors(index_case))
    rad2 = set()
    for n in rad1:
        rad2.update(G.neighbors(n))
    rad2.discard(index_case)
    rad2 = rad2 - rad1
    
    # Keep only the component connected to index case within radius 3 for visual clarity
    rad3 = set()
    for n in rad2:
        rad3.update(G.neighbors(n))
    rad3 = rad3 - rad2 - rad1
    rad3.discard(index_case)
    
    nodes_to_keep = {index_case} | rad1 | rad2 | rad3
    G_sub = G.subgraph(nodes_to_keep)
    
    pos = nx.spring_layout(G_sub, seed=42, k=0.15)
    
    # Draw edges
    nx.draw_networkx_edges(G_sub, pos, ax=ax2, alpha=0.2, edge_color="gray")
    
    # Draw nodes
    nx.draw_networkx_nodes(G_sub, pos, nodelist=list(rad3), node_color="#ecf0f1", node_size=30, ax=ax2)
    nx.draw_networkx_nodes(G_sub, pos, nodelist=list(rad2), node_color="#f1c40f", node_size=50, ax=ax2, label="Radius 2 (Contacts of Contacts)")
    nx.draw_networkx_nodes(G_sub, pos, nodelist=list(rad1), node_color="#e67e22", node_size=100, ax=ax2, label="Radius 1 (Direct Contacts)")
    nx.draw_networkx_nodes(G_sub, pos, nodelist=[index_case], node_color="#c0392b", node_size=300, ax=ax2, label="Index Case")
    
    ax2.legend(loc="lower left", frameon=False, fontsize=12)
    
    # ---------------------------------------------------------
    # Panel C: Calibration
    # ---------------------------------------------------------
    ax3 = plt.subplot(gs[2])
    ax3.set_title("C", loc="left", fontsize=24, fontweight="bold")
    
    import matplotlib.ticker as ticker
    
    ax3.bar(df_daily['Date'], cases_inc, color="#5DADE2", width=1.0, alpha=0.6, edgecolor='black', linewidth=0.5, label="Confirmed Cases")
    
    # Add 7-day moving average for cases
    cases_inc_ma = pd.Series(cases_inc).rolling(window=7, min_periods=1).mean().values
    ax3.plot(df_daily['Date'], cases_inc_ma, color="#2980B9", lw=2.5, label="Cases (7-day MA)")
    
    ax3.bar(df_daily_deaths['Date'], -deaths_inc, color="#5D6D7E", width=1.0, edgecolor='black', linewidth=0.5, label="Confirmed Deaths")
    ax3.set_ylabel("Daily Incidence (Cases $\\uparrow$, Deaths $\\downarrow$)", fontsize=14)
    ax3.axhline(0, color='black', lw=1.5)
    
    # Format y-axis to show absolute values (remove negative signs from death axis)
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
    
    ax3.legend(loc="upper left", frameon=False, fontsize=12)
    
    ax3.set_xlabel("Date (2026)", fontsize=14)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax3_twin = ax3.twinx()
    
    # Load EpiNow2 Rt
    epinow_path = 'results/epinow_rt.csv'
    if not os.path.exists(epinow_path):
        epinow_path = '../results/epinow_rt.csv'
    if os.path.exists(epinow_path):
        rt_df = pd.read_csv(epinow_path)
        rt_df['date'] = pd.to_datetime(rt_df['date'])
        
        rt_len = min(len(rt_df), len(df_daily['Date']))
        
        ax3_twin.fill_between(rt_df['date'][:rt_len], 
                              rt_df['lower_90'][:rt_len], 
                              rt_df['upper_90'][:rt_len], 
                              color="#C0392B", alpha=0.2, label="90% CrI")

        ax3_twin.plot(rt_df['date'][:rt_len], rt_df['median'][:rt_len], color="#C0392B", lw=3, label="Estimated $R_t$")
        
    ax3_twin.axhline(1.0, color='gray', linestyle='--', alpha=0.5, lw=2)
    ax3_twin.set_ylabel("Effective Reproduction Number ($R_t$)", fontsize=14, color="#C0392B")
    ax3_twin.tick_params(axis='y', labelcolor="#C0392B")
    
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3_twin.spines['top'].set_visible(False)
    
    max_cases = max(cases_inc.max(), deaths_inc.max())
    ax3.set_ylim(-max_cases * 1.1, max_cases * 1.1)
    
    if os.path.exists(epinow_path):
        ax3_twin.set_ylim(0, max(6.0, np.nanmax(rt_df['upper_90'])*1.1))
        
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"fig1_model_architecture_{timestamp}.png"
    img_path = figure_path("polished") / img_name
    os.makedirs(figure_path("polished"), exist_ok=True)
    
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved Figure 1 to {img_path}")
    
    # Also save to Dropbox for the user
    dropbox_dir = "/Users/jasonandrews/Library/CloudStorage/Dropbox/Isaac-Jason/_Ebola/manuscript/Figures_v43_high_res"
    plt.savefig(f"{dropbox_dir}/Figure_1_model_structure_calibration.png", dpi=300, facecolor='white')
    plt.savefig(f"{dropbox_dir}/Figure_1_model_structure_calibration.pdf", dpi=300, facecolor='white', format='pdf')
    print(f"Saved Figure 1 to Dropbox {dropbox_dir}")
    
    # Update walkthrough
    walkthrough_path = Path("figures/walkthrough.md")
    if walkthrough_path.exists():
        import re
        with open(walkthrough_path, "r") as f:
            content = f.read()
        content = re.sub(r'!\[Figure 1\].*?\.png\)', f'![Figure 1](polished/{img_name})', content)
        with open(walkthrough_path, "w") as f:
            f.write(content)
        print("Updated walkthrough.md")

if __name__ == "__main__":
    main()
