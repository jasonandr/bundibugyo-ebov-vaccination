import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

os.makedirs("figures/new", exist_ok=True)
raw = pd.read_csv("data_and_results/final_high_replicate_raw.csv")

color_r1 = '#2c3e50'
color_r2 = '#e67e22'
color_opt = '#27ae60'
color_base = '#7f8c8d'

def get_averted(df, b_map, col="deaths_percent"):
    def averted(row):
        b = b_map.get(row['seed'], np.nan)
        if pd.isna(b) or b == 0.0:
            return np.nan
        val = (b - row[col]) / b * 100.0
        return max(-100.0, val)
    
    df = df.copy()
    if col == "deaths_percent":
        df["Deaths_Averted"] = df.apply(averted, axis=1)
    else:
        df["Cases_Averted"] = df.apply(averted, axis=1)
    
    df = df.dropna(subset=[f"{'Deaths' if col == 'deaths_percent' else 'Cases'}_Averted"])
    return df

def plot_strip_box(ax, df, x, y, palette=None, ylabel="", ylim=(0, 80)):
    # By omitting hue entirely, the violin and stripplot align perfectly on the x ticks
    sns.violinplot(data=df, x=x, y=y, palette=palette, ax=ax, inner=None, cut=0)
    strip_df = pd.concat([group.sample(min(len(group), 400), random_state=42) for _, group in df.groupby(x, observed=False)])
    sns.stripplot(data=strip_df, x=x, y=y, palette=palette, alpha=0.2, size=2.5, ax=ax, jitter=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)

# --- Figure 2: Base Case ---
baseline_df = raw[raw["scenario"] == "fig2_base"]
baseline_novax = baseline_df[baseline_df["level"] == "no_vax"]
baseline_map = baseline_novax.set_index('seed')['deaths_percent'].to_dict()
cases_map = baseline_novax.set_index('seed')['cases_percent'].to_dict()

df2 = raw[(raw["scenario"] == "fig2_base") & (raw["level"] != "no_vax")].copy()
df2["Base_Cases"] = df2["seed"].map(cases_map)
df2["Base_Deaths"] = df2["seed"].map(baseline_map)
df2["Cases_Averted_Abs"] = (df2["Base_Cases"] - df2["cases_percent"]) * 1000  # N=100,000
df2["Deaths_Averted_Abs"] = (df2["Base_Deaths"] - df2["deaths_percent"]) * 1000
df2["Group"] = df2["level"].map({"radius1": "Radius 1", "radius2": "Radius 2"})

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
sns.scatterplot(data=df2, x="Cases_Averted_Abs", y="Deaths_Averted_Abs", hue="Group", palette={'Radius 1': color_r1, 'Radius 2': color_r2}, alpha=0.5, s=20, ax=ax, edgecolor='none')
ax.set_xlabel("Cases Averted (Absolute Count)")
ax.set_ylabel("Deaths Averted (Absolute Count)")
ax.set_title("Paired Delta: Absolute Cases vs Deaths Averted")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add CFR reference line
x_vals = np.array([0, df2["Cases_Averted_Abs"].max()])
ax.plot(x_vals, x_vals * 0.454, color='gray', linestyle='--', alpha=0.7, label='Expected Baseline CFR (45.4%)')
ax.legend(frameon=False)

plt.tight_layout()
fig2_path = f"figures/new/fig2_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
plt.savefig(fig2_path)
plt.close()

# --- Figure 3: Bio Sensitivity ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
df3a = get_averted(raw[raw["scenario"] == "fig3a_ve"], baseline_map)
df3b = get_averted(raw[raw["scenario"] == "fig3b_pep"], baseline_map)

pal_3a = sns.color_palette("Oranges", 4)
pal_3b = sns.color_palette("Greens", 3)

plot_strip_box(axes[0], df3a, "level", "Deaths_Averted", palette=pal_3a, ylabel="Deaths Averted (%)", ylim=(0, 80))
axes[0].set_title("A. Vaccine Efficacy")
plot_strip_box(axes[1], df3b, "level", "Deaths_Averted", palette=pal_3b, ylabel="", ylim=(0, 80))
axes[1].set_title("B. PEP Assumptions")
plt.tight_layout()
fig3_path = f"figures/new/fig3_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
plt.savefig(fig3_path)
plt.close()

# --- Figure 4: Operational Contour Pairs ---
df4_raw = raw[raw["scenario"] == "fig4c_heatmap"].copy()
df4_raw["delay"] = df4_raw["level"].apply(lambda x: float(x.split('_')[2]))
df4_raw["det"] = df4_raw["level"].apply(lambda x: float(x.split('_')[1]))

df4_base = df4_raw[df4_raw["radius"] == 0].groupby(["det", "delay"])["deaths_percent"].median().reset_index()
df4_vax = df4_raw[df4_raw["radius"] == 2].groupby(["det", "delay"])["deaths_percent"].median().reset_index()

unmitigated_median = baseline_novax["deaths_percent"].median()
merged = pd.merge(df4_base, df4_vax, on=["det", "delay"], suffixes=("_base", "_vax"))
merged["Ops_Averted"] = (unmitigated_median - merged["deaths_percent_base"]) / unmitigated_median * 100
merged["Vax_Added"] = (merged["deaths_percent_base"] - merged["deaths_percent_vax"]) / merged["deaths_percent_base"] * 100

vax_ops = merged.pivot(index="det", columns="delay", values="Ops_Averted")
vax_added = merged.pivot(index="det", columns="delay", values="Vax_Added")
X, Y = np.meshgrid(vax_ops.columns, vax_ops.index)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

contour1 = axes[0].contourf(X, Y, vax_ops.values, levels=15, cmap='viridis')
plt.colorbar(contour1, ax=axes[0], label="Median Deaths Averted (%)")
axes[0].contour(X, Y, vax_ops.values, levels=15, colors='white', alpha=0.3, linewidths=0.5)
axes[0].set_title("A. Impact of Operations Alone (No Vaccine)")
axes[0].set_xlabel("Detection Delay (Days)")
axes[0].set_ylabel("Detection Rate")

contour2 = axes[1].contourf(X, Y, vax_added.values, levels=15, cmap='plasma')
plt.colorbar(contour2, ax=axes[1], label="Median Deaths Averted (%)")
axes[1].contour(X, Y, vax_added.values, levels=15, colors='white', alpha=0.3, linewidths=0.5)
axes[1].set_title("B. Additional Benefit of Adding Vaccine")
axes[1].set_xlabel("Detection Delay (Days)")
axes[1].set_ylabel("Detection Rate")

plt.tight_layout()
fig4_path = f"figures/new/fig4_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
plt.savefig(fig4_path)
plt.close()

# --- Figure 5: Timing of Immunogenicity Curves and Impact (Formerly Fig 6) ---
days = np.linspace(0, 21, 500)
def sigmoidal(t, d0=10.0, k=0.5):
    return 1.0 / (1.0 + np.exp(-k * (t - d0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Panel A: Curves
ax = axes[0]
ax.plot(days, sigmoidal(days, d0=10.0, k=0.5), color=color_opt, linewidth=3, label="Base Case (Sigmoidal, 10d)")
ax.plot(days, sigmoidal(days, d0=5.0, k=0.5), color='#3498db', linewidth=2, linestyle='--', label="Fast (Sigmoidal, 5d)")
ax.plot(days, sigmoidal(days, d0=14.0, k=0.5), color='#e74c3c', linewidth=2, linestyle=':', label="Slow (Sigmoidal, 14d)")
ax.step(days, np.where(days >= 10, 1.0, 0.0), color='#9b59b6', linewidth=2, where='post', label="10-day Step Function")

ax.set_xlabel("Days Since Vaccination")
ax.set_ylabel("Proportion of Max Efficacy Reached")
ax.set_title("A. Timing of Immunogenicity Assumptions")
ax.set_xlim(0, 21)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)

# Panel B: Impact (Point Plot instead of Violin)
ax2 = axes[1]
df6 = get_averted(raw[raw["scenario"] == "fig3c_onset"], baseline_map)
pal_6 = sns.color_palette("Purples", 4)
sns.pointplot(data=df6, x="level", y="Deaths_Averted", errorbar=("pi", 75), palette=pal_6, ax=ax2, capsize=0.1, markers="o", linestyle="none")
ax2.set_ylabel("Deaths Averted (%)")
ax2.set_xlabel("")
ax2.set_title("B. Impact on Deaths Averted (Mean ± 75% IQR)")
ax2.set_ylim(0, 80)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig5_path = f"figures/new/fig5_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
plt.savefig(fig5_path)
plt.close()

print(f"FIG2={fig2_path}")
print(f"FIG3={fig3_path}")
print(f"FIG4={fig4_path}")
print(f"FIG5={fig5_path}")
