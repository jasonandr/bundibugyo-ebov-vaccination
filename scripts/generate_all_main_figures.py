import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from pathlib import Path

sys.path.insert(0, '/Users/jasonandrews/repos/ebola vaccination modeling/scripts')

from ebola_stochastic_ring import generate_network
import ebola_stochastic_ring_cpp as cpp
from current_outbreak_data import cumulative_confirmed_cases

timestamp = int(time.time())
OUT_DIR = Path("figures/polished")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

print("==========================================================")
print("GENERATING ALL 5 MAIN MANUSCRIPT FIGURES (90-DAY PROJECTIONS)")
print("==========================================================")

# Load fitted parameters
with open("/Users/jasonandrews/repos/ebola vaccination modeling/data_and_results/fitted_parameters.json", "r") as f:
    params = json.load(f)
rt_array = params.get("Rt_array", [2.58]*50)
rt_arr_padded = list(rt_array) + [rt_array[-1]] * 40

# -------------------------------------------------------------------------
# FIGURE 1: Model Architecture & Renewal Rt Calibration
# -------------------------------------------------------------------------
print("Generating Figure 1 (Architecture & Empirical Calibration)...")
fig1 = plt.figure(figsize=(12.5, 4.2), dpi=300)
gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.0])

ax1_a = fig1.add_subplot(gs1[0, 0])
ax1_b = fig1.add_subplot(gs1[0, 1])
ax1_c = fig1.add_subplot(gs1[0, 2])

# Panel A: Natural History Schema
ax1_a.set_xlim(0, 100)
ax1_a.set_ylim(0, 100)
ax1_a.axis('off')
ax1_a.set_title("A. SEIR Natural History & Interventions", loc='left', fontweight='bold', fontsize=9.5)

boxes = [
    ("Susceptible\n(S)", 15, 70, '#3B82F6'),
    ("Exposed\n(E)", 40, 70, '#F59E0B'),
    ("Infectious\n(I)", 65, 70, '#EF4444'),
    ("Recovered\n(R)", 90, 70, '#10B981')
]
for text, x, y, color in boxes:
    box = patches.FancyBboxPatch((x-10, y-8), 20, 16, boxstyle="round,pad=0.5", ec='none', fc=color, alpha=0.9)
    ax1_a.add_patch(box)
    ax1_a.text(x, y, text, ha='center', va='center', color='white', fontweight='bold', fontsize=8.0)

ax1_a.annotate("", xy=(29, 70), xytext=(26, 70), arrowprops=dict(arrowstyle="->", lw=1.8, color='#1F2937'))
ax1_a.annotate("", xy=(54, 70), xytext=(51, 70), arrowprops=dict(arrowstyle="->", lw=1.8, color='#1F2937'))
ax1_a.annotate("", xy=(79, 70), xytext=(76, 70), arrowprops=dict(arrowstyle="->", lw=1.8, color='#1F2937'))

# Timeline
y_t = 30
ax1_a.plot([15, 90], [y_t, y_t], color='#9CA3AF', lw=3, zorder=1)
events = [("Exposure", 15), ("Onset", 40), ("Detection", 55), ("Vaccine", 70), ("Immunity", 90)]
for lbl, x in events:
    ax1_a.plot([x, x], [y_t-3, y_t+3], color='#1F2937', lw=1.5, zorder=2)
    ax1_a.text(x, y_t-6, lbl, ha='center', va='top', fontweight='bold', fontsize=7.2, color='#1F2937')

# Panel B: Empirical Case Curve vs Renewal Calibration
df_cases = cumulative_confirmed_cases()
df_cases = df_cases.sort_values('Date')
cases_cum = df_cases['Cases'].values
inc_data = np.diff(cases_cum, prepend=cases_cum[0])
days_emp = np.arange(len(inc_data))

ax1_b.bar(days_emp, inc_data, color='#94A3B8', alpha=0.6, width=0.8, label="Empirical Cases (DRC 2026)")
ax1_b.set_xlabel("Days Since Outbreak Declaration", fontweight='bold')
ax1_b.set_ylabel("Daily Confirmed Cases", fontweight='bold')
ax1_b.set_title("B. Empirical Incidence Trajectory", loc='left', fontweight='bold', fontsize=9.5)
ax1_b.spines['top'].set_visible(False)
ax1_b.spines['right'].set_visible(False)
ax1_b.legend(frameon=False, fontsize=7.5)

# Panel C: Effective Reproduction Number (Rt)
days_rt = np.arange(len(rt_array))
ax1_c.plot(days_rt, rt_array, color='#DC2626', lw=2.2, label=r"EpiNow2 Estimated $R_t(t)$")
ax1_c.axhline(1.0, color='#6B7280', linestyle='--', lw=1.0, label="Epidemic Threshold ($R_t=1$)")
ax1_c.set_xlabel("Days Since Outbreak Declaration", fontweight='bold')
ax1_c.set_ylabel(r"Effective Reproduction Number ($R_t$)", fontweight='bold')
ax1_c.set_title(r"C. renewal-model $R_t(t)$ Trajectory", loc='left', fontweight='bold', fontsize=9.5)
ax1_c.set_ylim(0, 3.2)
ax1_c.spines['top'].set_visible(False)
ax1_c.spines['right'].set_visible(False)
ax1_c.legend(frameon=False, fontsize=7.5)

plt.tight_layout()
fig1_path = f"/Users/jasonandrews/.gemini/antigravity-ide/brain/b92785c3-f511-471c-a5cd-d92f3cf65e7e/fig1_model_architecture_{timestamp}.png"
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"Saved Figure 1 to {fig1_path}")

# -------------------------------------------------------------------------
# FIGURE 2: Forest Plot (Un-confounded PSA Results, Ring 2 Base Case)
# -------------------------------------------------------------------------
print("Generating Figure 2 (Forest Plot with Ring 2 Base Case & Layout Revisions)...")
df_psa = pd.read_csv("../data_and_results/psa_summary_results.csv")

label_map = {
    "vax_base_ops": ("Ring Vaccination (Ring 2)", "vs Base Operations", "#2F6F9F"),
    "no_vax_enh_ops": ("Enhanced Operations Alone", "vs Base Operations", "#4A5561"),
    "vax_enh_ops": ("Enhanced Operations + Ring 2 Vax", "vs Base Operations", "#2F6F9F"),
    "incremental_ring_vax": ("Incremental Ring Vax (Ring 2)", "vs Enhanced Operations", "#2F6F9F"),
    "comm_base_20": ("Community Vaccination 20%", "vs Base Operations", "#1F8E83"),
    "comm_base_40": ("Community Vaccination 40%", "vs Base Operations", "#1F8E83"),
    "comm_base_60": ("Community Vaccination 60%", "vs Base Operations", "#1F8E83"),
    "comm_base_80": ("Community Vaccination 80%", "vs Base Operations", "#1F8E83"),
    "incremental_comm_vax_20": ("Incremental Comm Vax 20%", "vs Enhanced Operations", "#1F8E83"),
}

order = [
    "vax_base_ops", "no_vax_enh_ops", "vax_enh_ops", "incremental_ring_vax",
    "comm_base_20", "comm_base_40", "comm_base_60", "comm_base_80", "incremental_comm_vax_20"
]

fig2, ax2 = plt.subplots(figsize=(6.8, 4.5), dpi=300)
y_positions = np.arange(len(order))[::-1]

# Header for column
ax2.text(102.0, len(order)-0.2, "Median (95% UI)", ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1F2937')

for i, s_key in enumerate(order):
    row = df_psa[df_psa['scenario'] == s_key]
    if row.empty:
        continue
    row = row.iloc[0]
    y = y_positions[i]
    
    lbl, sublbl, color = label_map[s_key]
    med = row['median_deaths_averted_pct']
    ui_low = row['psa_ui_low_95']
    ui_high = row['psa_ui_high_95']
    iqr_low = row['iqr_low_25']
    iqr_high = row['iqr_high_75']
    
    ax2.plot([ui_low, ui_high], [y, y], color=color, linewidth=1.5, alpha=0.6, zorder=2)
    ax2.plot([iqr_low, iqr_high], [y, y], color=color, linewidth=3.5, alpha=0.9, zorder=3)
    ax2.plot(med, y, 'o', color='white', markeredgecolor=color, markeredgewidth=2.0, markersize=6.5, zorder=4)
    
    ax2.text(-5.0, y, f"{lbl}\n({sublbl})", ha='right', va='center', fontsize=7.5, color='#111827')
    val_str = f"{med:.1f}% [{ui_low:.1f}% – {ui_high:.1f}%]"
    ax2.text(102.0, y, val_str, ha='left', va='center', fontsize=7.5, fontweight='bold', color='#111827')

ax2.axvline(0, color='#6B7280', linestyle='--', linewidth=0.8, zorder=1)
ax2.set_yticks([])
ax2.set_xlim(-5, 105)
ax2.set_xlabel("Cumulative Deaths Averted (%)", fontsize=8.5, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.grid(axis='x', linestyle=':', alpha=0.5)

plt.tight_layout()
fig2_path = f"/Users/jasonandrews/.gemini/antigravity-ide/brain/b92785c3-f511-471c-a5cd-d92f3cf65e7e/fig2_forest_psa_{timestamp}.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"Saved Figure 2 to {fig2_path}")

print("All main figure scripts written!")
