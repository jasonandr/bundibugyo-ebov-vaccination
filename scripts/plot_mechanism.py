import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import os
import ebola_stochastic_ring_cpp as sim

# 1. Setup Environment
OUT_DIR = "figures/new_analyses"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
})

import ebola_stochastic_ring as py_sim
# 2. Network and params
print("Loading network...")
N = 100000
nx_G = py_sim.generate_network(N)
_G = [list(nx_G.neighbors(i)) for i in range(N)]

with open("data_and_results/rt_calibrated_tau_array.json") as f:
    TAU_ARRAY = json.load(f)["tau_array"]

# 3. Simulate Mechanism
def run_scenario(strategy_name, comm_cov, ring_rad, n_sims=500):
    all_res = []
    for i in range(n_sims):
        enh_reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
        enh_tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76
        
        # Baseline
        base_res = sim.simulate_mechanism_cpp(
            N, _G, TAU_ARRAY, 0.25, 9.7, 7.0, 2, 0.0, 10.0, 0.65, enh_reporting, 1.0, 3.0, 1.0,
            2000, 15, -1, 0.454, 0.227, 5, 5, 90, 0.0, False, 1.0, False, enh_tracing, -1.0,
            0.5, 5.0, 0.8, 1.0, True, True, 0.0, 0, 0.0, i
        )
        
        # Strategy
        res = sim.simulate_mechanism_cpp(
            N, _G, TAU_ARRAY, 0.25, 9.7, 7.0, ring_rad, 0.40, 10.0, 0.65, enh_reporting, 1.0, 3.0, 1.0,
            2000, 15, 0 if comm_cov > 0 else -1, 0.454, 0.227, 5, 5, 90, 0.0, False, 1.0, False, enh_tracing, -1.0,
            0.5, 5.0, 0.8, 1.0, True, True, comm_cov, 2 if comm_cov > 0 else 0, 0.0, i
        )
        
        # Calculate derived metrics
        infected = np.array(res["total_infected"])
        vaccines = np.array(res["total_vaccines"])
        
        exposure = np.array(res["exposure_time"])
        vax = np.array(res["vaccination_time"])
        onset = np.array(res["onset_time"])
        state = np.array(res["state_at_vaccination"])
        
        # Filter to those who were both exposed and vaccinated
        valid = (exposure >= 0) & (vax >= 0)
        exposure_valid = exposure[valid]
        vax_valid = vax[valid]
        onset_valid = onset[valid]
        state_valid = state[valid]
        
        # Diff
        diff = vax_valid - exposure_valid
        
        # Decompose Deaths Averted
        base_cases = base_res["total_infected"]
        base_deaths = base_res["total_deaths"]
        strat_cases = res["total_infected"]
        strat_deaths = res["total_deaths"]
        
        deaths_averted = base_deaths - strat_deaths
        cases_averted = base_cases - strat_cases
        prophylactic_averted = cases_averted * 0.454
        pep_averted = deaths_averted - prophylactic_averted
        
        all_res.append({
            "Strategy": strategy_name,
            "Diffs": diff,
            "States": state_valid,
            "Prophylactic": prophylactic_averted,
            "PEP": pep_averted,
            "Deaths_Averted": deaths_averted
        })
    return all_res

print("Running simulations...")
# We use a modest N like 200 just to generate enough distributions without taking forever
res_ring = run_scenario("Radius 2 Ring", 0.0, 2, n_sims=300)
res_comm = run_scenario("Community Vax 40%", 0.40, 2, n_sims=300)

all_diffs = []
all_states = []
all_decomp = []

for r in res_ring + res_comm:
    strat = r["Strategy"]
    for d in r["Diffs"]:
        all_diffs.append({"Strategy": strat, "Vaccination - Exposure (Days)": d})
    for s in r["States"]:
        if s == 0: val = "Susceptible"
        elif s == 1: val = "Exposed"
        else: val = "Symptomatic"
        all_states.append({"Strategy": strat, "State": val})
        
    all_decomp.append({
        "Strategy": strat,
        "Prophylactic (Transmission Block)": r["Prophylactic"],
        "PEP (Mortality Reduction)": max(0, r["PEP"])
    })

df_diffs = pd.DataFrame(all_diffs)
df_states = pd.DataFrame(all_states)
df_decomp = pd.DataFrame(all_decomp)

# 4. Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
axes = axes.flatten()

# Panel A: Distribution of Time from Exposure to Vaccination
sns.kdeplot(
    data=df_diffs,
    x="Vaccination - Exposure (Days)",
    hue="Strategy",
    common_norm=False,
    fill=True,
    alpha=0.3,
    ax=axes[0],
    palette=["#4f6d7a", "#2a9d8f"],
    clip=(-30, 30)
)
axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title("A. Timing of vaccination relative to exposure", loc="left")
axes[0].set_xlabel("Days (Negative = Vaccinated before exposure)")

# Panel B: Fraction vaccinated by state
state_counts = df_states.groupby(["Strategy", "State"]).size().reset_index(name="Count")
state_totals = state_counts.groupby("Strategy")["Count"].sum().reset_index(name="Total")
state_counts = state_counts.merge(state_totals, on="Strategy")
state_counts["Fraction"] = state_counts["Count"] / state_counts["Total"]

sns.barplot(
    data=state_counts,
    x="Strategy",
    y="Fraction",
    hue="State",
    ax=axes[1],
    palette="viridis"
)
axes[1].set_title("B. Disease state at time of vaccination", loc="left")
axes[1].set_ylabel("Fraction of vaccinated eventual cases")

# Panel C: Fraction with meaningful immune/PEP benefit
# Meaningful benefit = vaccinated > 5 days before onset
df_diffs["Benefit"] = df_diffs["Vaccination - Exposure (Days)"] < 5 # Assuming ~10 day incubation, this means vax >5 days before onset
benefit_frac = df_diffs.groupby("Strategy")["Benefit"].mean().reset_index()
sns.barplot(
    data=benefit_frac,
    x="Strategy",
    y="Benefit",
    hue="Strategy",
    ax=axes[2],
    palette=["#4f6d7a", "#2a9d8f"],
    legend=False
)
axes[2].set_title("C. Fraction vaccinated early enough for strong PEP/Immunity", loc="left")
axes[2].set_ylabel("Fraction (Vax >5 days before expected onset)")

# Panel D: Deaths averted decomposed
decomp_melt = df_decomp.melt(id_vars="Strategy", value_vars=["Prophylactic (Transmission Block)", "PEP (Mortality Reduction)"], var_name="Mechanism", value_name="Deaths Averted")
sns.barplot(
    data=decomp_melt,
    x="Strategy",
    y="Deaths Averted",
    hue="Mechanism",
    ax=axes[3],
    palette="magma"
)
axes[3].set_title("D. Mechanism of mortality reduction", loc="left")
axes[3].set_ylabel("Deaths averted per network")

fig.tight_layout()
path = f"{OUT_DIR}/fig4_mechanism_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
plt.savefig(path)
print(f"MECHANISM={path}")
