import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import datetime
import os
import matplotlib.gridspec as gridspec
from ebola_stochastic_ring import generate_network
from paths import result_path
from current_outbreak_data import cumulative_confirmed_cases

print("Loading 2026 parameters and empirical data...")
with open(result_path("fitted_parameters.json"), "r") as f:
    params = json.load(f)
rt_array = params['Rt_array']

# Load 2026 Daily Data (Interpolated from cumulative to fill gaps)
df = cumulative_confirmed_cases()
df = df.sort_values('Date')

cases_raw = df['Cases'].values.copy()
for i in range(len(cases_raw)-2, -1, -1):
    if cases_raw[i] > cases_raw[i+1]:
        cases_raw[i] = cases_raw[i+1]
df['Cases'] = cases_raw

date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df_daily = pd.DataFrame({'Date': date_range})
df_daily = pd.merge(df_daily, df, on='Date', how='left')
df_daily['Cases'] = df_daily['Cases'].interpolate(method='linear')

cases_cum = df_daily['Cases'].values
inc_data = np.diff(cases_cum, prepend=cases_cum[0])
t_data_dates = df_daily['Date']

# Setup 1x3 Layout
fig = plt.figure(figsize=(18, 5.5), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

import matplotlib.patches as patches

# ==========================================
# Panel A: SEIR Model Schema & Timeline
# ==========================================
print("Generating Panel A (SEIR Schema)...")
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.axis('off')
# ax1.set_title("A. Natural History & Intervention Timeline", loc='left', fontsize=14, fontweight='bold', pad=15)
ax1.set_title('A', loc='left', size=18, weight='bold')

# Draw SEIR Blocks
y_seir = 70
boxes = [
    ("Susceptible\n(S)", 15, y_seir, '#4A90E2'),
    ("Exposed\n(E)", 40, y_seir, '#F39C12'),
    ("Infectious\n(I)", 65, y_seir, '#E74C3C'),
    ("Recovered\n(R)", 90, y_seir, '#7F8C8D')
]

for text, x, y, color in boxes:
    # Width 20, Height 14
    box = patches.FancyBboxPatch((x-10, y-7), 20, 14, boxstyle="round,pad=0.5", ec='white', fc=color, alpha=0.9)
    ax1.add_patch(box)
    ax1.text(x, y, text, ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
# SEIR Arrows (Space between boxes is 40-15=25. Box edge is 10. Space is 5)
ax1.annotate("", xy=(29, y_seir), xytext=(26, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
ax1.text(27.5, y_seir+2, r'$\lambda$', ha='center', va='bottom', fontsize=12)

ax1.annotate("", xy=(54, y_seir), xytext=(51, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
ax1.text(52.5, y_seir+2, r'$\sigma$', ha='center', va='bottom', fontsize=12)

ax1.annotate("", xy=(79, y_seir), xytext=(76, y_seir), arrowprops=dict(arrowstyle="->", lw=2, color='#2C3E50'))
ax1.text(77.5, y_seir+2, r'$\gamma$', ha='center', va='bottom', fontsize=12)

# Draw Timeline
y_time = 30
ax1.plot([15, 90], [y_time, y_time], color='#BDC3C7', lw=4, zorder=1, solid_capstyle='butt')

events = [
    ("Exposure", 15, ""),
    ("Onset", 40, ""),
    ("Detection", 55, ""),
    ("Vaccinate", 70, ""),
    ("Immunity", 90, "")
]

# Tick marks
for label, x, desc in events:
    ax1.plot([x, x], [y_time-3, y_time+3], color='#2C3E50', lw=2, zorder=2)
    ax1.text(x, y_time-5, label, ha='center', va='top', fontweight='bold', fontsize=9, color='#2C3E50')
    
# Interval brackets/text
ax1.annotate("", xy=(15, y_time+4), xytext=(40, y_time+4), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#7F8C8D'))
ax1.text(27.5, y_time+6, "Incubation\nPeriod", ha='center', va='bottom', fontsize=8, color='#7F8C8D')

ax1.annotate("", xy=(40, y_time+4), xytext=(55, y_time+4), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#7F8C8D'))
ax1.text(47.5, y_time+6, "Reporting\nDelay", ha='center', va='bottom', fontsize=8, color='#7F8C8D')

ax1.annotate("", xy=(55, y_time+4), xytext=(70, y_time+4), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#F39C12'))
ax1.text(62.5, y_time+6, "Tracing\nDelay", ha='center', va='bottom', fontsize=8, color='#F39C12', fontweight='bold')

ax1.annotate("", xy=(70, y_time+4), xytext=(90, y_time+4), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#27AE60'))
ax1.text(80, y_time+6, "Immune\nDelay", ha='center', va='bottom', fontsize=8, color='#27AE60', fontweight='bold')

# Connecting dashed line from SEIR to Timeline
ax1.plot([40, 40], [y_seir-8, y_time+12], ls=':', color='#BDC3C7', lw=1.5)
ax1.plot([15, 15], [y_seir-8, y_time+12], ls=':', color='#BDC3C7', lw=1.5)

# ==========================================
# Panel B: Ring Vaccination Schema
# ==========================================
print("Generating Panel B (Ring Schema)...")
N_plot = 400
G = generate_network(N_plot)
degrees = dict(G.degree())

# Find a high degree node to be the index case
index_case = max(degrees, key=degrees.get)
radius_1 = list(G.neighbors(index_case))

radius_2 = set()
for r1 in radius_1:
    radius_2.update(G.neighbors(r1))
radius_2.discard(index_case)
radius_2.difference_update(radius_1)
radius_2 = list(radius_2)

# Create subgraph of just the ring to layout nicely
sub_nodes = [index_case] + radius_1 + radius_2
subG = G.subgraph(sub_nodes)
pos_sub = nx.spring_layout(subG, k=0.3, seed=10)

nx.draw_networkx_edges(subG, pos_sub, ax=ax2, alpha=0.3, edge_color='gray')

# Draw layers with clear categorical colors
nx.draw_networkx_nodes(subG, pos_sub, nodelist=radius_2, ax=ax2, node_size=60, node_color='#f1c40f', edgecolors='white', label='Radius 2 (Contacts of Contacts)')
nx.draw_networkx_nodes(subG, pos_sub, nodelist=radius_1, ax=ax2, node_size=100, node_color='#e67e22', edgecolors='white', label='Radius 1 (Direct Contacts)')
nx.draw_networkx_nodes(subG, pos_sub, nodelist=[index_case], ax=ax2, node_size=200, node_color='#c0392b', edgecolors='white', label='Index Case')

# ax2.set_title("B. Reactive Ring Vaccination Logic", fontsize=14, fontweight='bold')
ax2.set_title('B', loc='left', size=18, weight='bold')
ax2.legend(loc='lower left', fontsize=9, frameon=False)
ax2.axis('off')

# ==========================================
# Panel C: Current Epidemic Calibration (2026)
# ==========================================
print("Generating Panel C (2026 Calibration)...")
color_cases = '#2980b9'
color_rt = '#c0392b'

ax3.bar(t_data_dates, inc_data, color=color_cases, alpha=0.8, width=0.8, edgecolor='none', label='Confirmed Cases')
ax3.set_xlabel("Date (2026)", fontsize=12)
ax3.set_ylabel("Daily Confirmed Cases", fontsize=12, color=color_cases)
ax3.tick_params(axis='y', labelcolor=color_cases)

import matplotlib.dates as mdates
max_date = df_daily['Date'].max() + pd.Timedelta(days=1)
ax3.set_xlim(pd.to_datetime(t_data_dates.min()) - pd.Timedelta(days=1), max_date)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax3_rt = ax3.twinx()
rt_dates = pd.date_range(start=t_data_dates.min(), periods=len(rt_array), freq='D')
ax3_rt.plot(rt_dates, rt_array, color=color_rt, linewidth=3, label='Estimated $R_t$')
ax3_rt.set_ylabel("Effective Reproduction Number ($R_t$)", fontsize=12, color=color_rt)
ax3_rt.tick_params(axis='y', labelcolor=color_rt)
ax3_rt.set_ylim(0, max(rt_array) * 1.2)
ax3_rt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

# ax3.set_title("C. Current Outbreak Calibration (2026)", fontsize=14, fontweight='bold')
ax3.set_title('C', loc='left', size=18, weight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3_rt.spines['top'].set_visible(False)

plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
png_filename = f"figures/fig1_model_architecture_{timestamp}.png"
pdf_filename = f"figures/fig1_model_architecture_{timestamp}.pdf"

plt.savefig(png_filename, bbox_inches='tight')
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
print(f"Saved Figure 1 to {png_filename} and .pdf")

walkthrough_path = "figures/walkthrough.md"
import re
if os.path.exists(walkthrough_path):
    with open(walkthrough_path, "r") as f: content = f.read()
    content = re.sub(r"!\[Figure 1\].*?\.png\)", f"![Figure 1]({png_filename})", content)
    with open(walkthrough_path, "w") as f: f.write(content)
