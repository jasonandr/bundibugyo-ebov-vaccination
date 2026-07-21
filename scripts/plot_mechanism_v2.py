import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import ebola_stochastic_ring as sim
import matplotlib

matplotlib.use('Agg')

# Strict categorical color palette requested by user
COLORS = {
    "No Vaccine": "#4B5563",         # Charcoal
    "Reactive Ring": "#4F6D7A",      # Blue-grey
    "Community Vax": "#1F9D8A",      # Teal
    "Hybrid": "#D96B4A"              # Coral
}

import multiprocessing as mp
from functools import partial

def run_single_mechanism(i, strat_name):
    # Re-generate network per worker to avoid serialization overhead
    G = sim.generate_network(10000)
    tau_array = np.linspace(0.12, 0.05, 91).tolist()
    enh_reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
    enh_tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76
    
    if strat_name == "Reactive Ring":
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.40, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=0.0,
            max_sim_time=90, seed=None, engine='cpp', return_mechanism=True
        )
    else:
        r = sim.simulate_ring_vaccination(
            G, initial_infected=5, rt_array=tau_array, ring_radius=2, baseline_tau=0.25,
            efficacy=0.40, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
            community_vax_coverage=0.40, community_vax_trigger=2, community_vax_delay=0.0,
            max_sim_time=90, seed=None, engine='cpp', return_mechanism=True
        )
        
    exposure = np.array(r.get("exposure_time", []))
    vax = np.array(r.get("vaccination_time", []))
    
    # We need valid pairs
    if len(exposure) == 0 or len(vax) == 0:
        return [], [], {"Strategy": strat_name, "Transmission prevention": 0, "Mortality benefit": 0}
        
    valid = (exposure >= 0) & (vax >= 0)
    exposure_valid = exposure[valid]
    vax_valid = vax[valid]
    diff = vax_valid - exposure_valid
    
    # Subsample to avoid KDE memory explosion
    if len(diff) > 20:
        indices = np.random.choice(len(diff), size=20, replace=False)
        diff = diff[indices]
        
    my_diffs = []
    my_states = []
    for d in diff:
        my_diffs.append({"Strategy": strat_name, "Vaccination - Exposure (Days)": d})
        if d < 0:
            state = "Susceptible"
        elif d <= 10:
            state = "Exposed/Incubating"
        elif d <= 21:
            state = "Symptomatic"
        else:
            state = "No longer relevant"
        my_states.append({"Strategy": strat_name, "State": state})
        
    baseline_total = 10000.0
    aborted = np.array(r.get("aborted_due_to_pep", []))
    pep_averted = np.sum(aborted)
    baseline_expected_deaths = 1500.0 
    total_deaths = r.get("total_deaths", 0)
    total_averted = max(0, baseline_expected_deaths - total_deaths)
    proph_averted = max(0, total_averted - pep_averted)
    
    my_decomp = {
        "Strategy": strat_name,
        "Transmission prevention": proph_averted / baseline_total * 100,
        "Mortality benefit": pep_averted / baseline_total * 100
    }
    
    return my_diffs, my_states, my_decomp

def extract_data_parallel(n_reps):
    all_diffs = []
    all_states = []
    all_decomp = []
    
    with mp.Pool(mp.cpu_count() - 1) as pool:
        # Ring
        func = partial(run_single_mechanism, strat_name="Reactive Ring")
        for d, s, decomp in pool.imap_unordered(func, range(n_reps), chunksize=50):
            all_diffs.extend(d)
            all_states.extend(s)
            all_decomp.append(decomp)
            
        # Comm
        func = partial(run_single_mechanism, strat_name="Community Vax")
        for d, s, decomp in pool.imap_unordered(func, range(n_reps), chunksize=50):
            all_diffs.extend(d)
            all_states.extend(s)
            all_decomp.append(decomp)
            
    return pd.DataFrame(all_diffs), pd.DataFrame(all_states), pd.DataFrame(all_decomp)

def plot_mechanism():
    df_diffs, df_states, df_decomp = extract_data_parallel(500)
    
    # Setup Figure Layout
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # Panel A: Top Row Full Width
    axA = fig.add_subplot(gs[0, :])
    # Panel B, C, D: Bottom Row
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])
    
    # --- PANEL A: Mirrored Density ---
    # We will use KDE plots overlaid
    sns.kdeplot(
        data=df_diffs[df_diffs["Strategy"] == "Community Vax"],
        x="Vaccination - Exposure (Days)",
        color=COLORS["Community Vax"],
        fill=True, alpha=0.4, ax=axA, label="Community vaccination", clip=(-30, 30)
    )
    sns.kdeplot(
        data=df_diffs[df_diffs["Strategy"] == "Reactive Ring"],
        x="Vaccination - Exposure (Days)",
        color=COLORS["Reactive Ring"],
        fill=True, alpha=0.4, ax=axA, label="Reactive ring vaccination", clip=(-30, 30)
    )
    
    # Annotations
    axA.axvline(0, color='black', linestyle='-', linewidth=1.5)
    axA.axvline(10, color='black', linestyle='--', linewidth=1.5)
    axA.axvspan(10, 30, color='grey', alpha=0.1)
    
    axA.text(-15, axA.get_ylim()[1]*0.9, "Before exposure", ha='center', fontsize=10)
    axA.text(5, axA.get_ylim()[1]*0.9, "Incubating", ha='center', fontsize=10)
    axA.text(20, axA.get_ylim()[1]*0.9, "Likely too late", ha='center', fontsize=10)
    
    # Direct labeling (no legend)
    axA.text(-10, axA.get_ylim()[1]*0.5, "Community\nvaccination", color=COLORS["Community Vax"], fontweight='bold')
    axA.text(12, axA.get_ylim()[1]*0.3, "Reactive ring\nvaccination", color=COLORS["Reactive Ring"], fontweight='bold')
    
    axA.set_title("A", loc="left", fontweight='bold', fontsize=14)
    axA.set_xlabel("Days vaccinated relative to exposure")
    axA.set_ylabel("Density")
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    
    # --- PANEL B: State at vaccination (100% Stacked Horizontal Bar) ---
    state_order = ["Susceptible", "Exposed/Incubating", "Symptomatic", "No longer relevant"]
    state_colors = ["#1F9D8A", "#F59E0B", "#EF4444", "#D1D5DB"] # Teal, Amber, Muted Red, Light Grey
    
    state_counts = df_states.groupby(["Strategy", "State"]).size().reset_index(name="Count")
    state_totals = state_counts.groupby("Strategy")["Count"].sum().reset_index(name="Total")
    state_counts = state_counts.merge(state_totals, on="Strategy")
    state_counts["Fraction"] = state_counts["Count"] / state_counts["Total"] * 100
    
    pivot = state_counts.pivot(index="Strategy", columns="State", values="Fraction").reindex(columns=state_order).fillna(0)
    
    y_pos = [0, 1]
    strategies = ["Community Vax", "Reactive Ring"]
    lefts = [0, 0]
    
    for i, state in enumerate(state_order):
        widths = [pivot.loc[s, state] if state in pivot.columns else 0 for s in strategies]
        axB.barh(y_pos, widths, left=lefts, color=state_colors[i], label=state, height=0.6)
        
        # Annotate
        for j, w in enumerate(widths):
            if w > 10:
                axB.text(lefts[j] + w/2, y_pos[j], f"{int(w)}%", ha='center', va='center', color='white' if i < 3 else 'black', fontweight='bold')
        
        lefts = [l + w for l, w in zip(lefts, widths)]
        
    axB.set_yticks(y_pos)
    axB.set_yticklabels(strategies)
    axB.set_xlabel("Vaccinated individuals (%)")
    axB.set_title("B", loc="left", fontweight='bold', fontsize=14)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    
    # Legend above Panel B
    axB.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2, frameon=False, fontsize=9)
    axB.grid(axis='x', linestyle='--', alpha=0.3)
    
    # --- PANEL C: Vaccinated early enough (Dot and interval) ---
    df_diffs["Benefit"] = (df_diffs["Vaccination - Exposure (Days)"] < 5).astype(int) * 100
    benefit_stats = df_diffs.groupby("Strategy")["Benefit"].agg(
        median='median',
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75)
    ).reindex(strategies)
    
    for i, strat in enumerate(strategies):
        axC.plot([benefit_stats.loc[strat, 'p25'], benefit_stats.loc[strat, 'p75']], [i, i], color=COLORS[strat], linewidth=3)
        axC.plot(benefit_stats.loc[strat, 'median'], i, 'o', color=COLORS[strat], markersize=10)
        axC.text(benefit_stats.loc[strat, 'p75'] + 5, i, f"{benefit_stats.loc[strat, 'median']:.0f}%", va='center', color=COLORS[strat], fontweight='bold')
        
    axC.set_yticks(y_pos)
    axC.set_yticklabels(["", ""])
    axC.set_xlabel("% vaccinated early enough for benefit")
    axC.set_title("C", loc="left", fontweight='bold', fontsize=14)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.set_xlim(0, 100)
    
    # --- PANEL D: Mechanism of mortality reduction ---
    decomp_mean = df_decomp.groupby("Strategy")[["Transmission prevention", "Mortality benefit"]].mean().reindex(strategies)
    
    axD.bar(strategies, decomp_mean["Transmission prevention"], color=COLORS["Community Vax"], label="Transmission prevention")
    axD.bar(strategies, decomp_mean["Mortality benefit"], bottom=decomp_mean["Transmission prevention"], color="#FCA5A5", label="Mortality benefit (PEP)") # Muted salmon
    
    for i, strat in enumerate(strategies):
        total = decomp_mean.loc[strat].sum()
        axD.text(i, total + 0.5, f"{total:.1f}", ha='center', fontweight='bold')
        
    axD.set_ylabel("Mortality reduction (percentage points)")
    axD.set_title("D", loc="left", fontweight='bold', fontsize=14)
    axD.spines['top'].set_visible(False)
    axD.spines['right'].set_visible(False)
    axD.legend(frameon=False, fontsize=9)
    
    plt.tight_layout()
    path = f"figures/new_analyses/fig5_mechanism_v2.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"FIG5={path}")

if __name__ == "__main__":
    plot_mechanism()
