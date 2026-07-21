import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from paths import result_path

def generate_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    color_r1 = '#2c3e50'
    raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

    # Baseline is enhanced_ops (70% reporting, 80% tracing)
    baseline_df = raw[(raw["scenario"] == "fig2_base") & (raw["level"] == "enhanced_ops")]
    baseline_map = baseline_df.set_index('seed')['deaths_percent'].to_dict()

    def deaths_averted(row):
        key = row['seed']
        baseline_deaths = baseline_map.get(key, 45.4) # Fallback just in case
        if baseline_deaths == 0:
            return 0.0
        return (baseline_deaths - row["deaths_percent"]) / baseline_deaths * 100.0

    def plot_boxes(ax, scenario, levels, label_map, xlabel):
        subset = raw.loc[raw["scenario"] == scenario].copy()
        subset = subset[subset["level"].astype(str).isin(levels)].copy()
        if len(subset) == 0:
            print(f"Warning: No data found for {scenario}")
            return
            
        subset["Averted"] = subset.apply(deaths_averted, axis=1)
        subset["Averted_display"] = subset["Averted"].clip(lower=0)
        subset["Group"] = subset["level"].astype(str).map(label_map)
        subset["Group"] = pd.Categorical(
            subset["Group"],
            categories=[label_map[l] for l in levels],
            ordered=True,
        )

        sns.violinplot(data=subset, x='Group', y='Averted_display', color=color_r1, ax=ax, cut=0, inner=None)

        strip_df = pd.concat(
            [group.sample(min(len(group), 600), random_state=20260630) for _, group in subset.groupby("Group", observed=False)],
            ignore_index=True,
        )
        
        # Darken color for strip
        hex_color = color_r1.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        strip_color = f"#{int(r*0.6):02x}{int(g*0.6):02x}{int(b*0.6):02x}"

        sns.stripplot(data=strip_df, x='Group', y='Averted_display', color=strip_color, alpha=0.3, size=2.5, ax=ax, jitter=True)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Deaths averted vs. Enhanced Ops (%)", fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(-5, 75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    def add_panel_label(ax, label):
        ax.set_title(label, loc='left', fontweight='bold', size=18)

    plot_boxes(axes[0], "fig3a_ve", ["ve_20", "ve_45", "ve_60", "ve_90"], {"ve_20": "20%", "ve_45": "45%", "ve_60": "60%", "ve_90": "90%"}, "Vaccine Efficacy")
    add_panel_label(axes[0], 'A. Vaccine Efficacy')

    plot_boxes(axes[1], "fig3b_pep", ["no_pep", "base_pep", "optimistic_pep"], {"no_pep": "No PEP", "base_pep": "Baseline PEP", "optimistic_pep": "Optimistic PEP"}, "Post-Exposure Prophylaxis")
    add_panel_label(axes[1], 'B. Post-Exposure Prophylaxis')

    plot_boxes(axes[2], "fig3c_onset", ["5_day", "10_day", "14_day"], {"5_day": "5 Days", "10_day": "10 Days", "14_day": "14 Days"}, "Time to 50% Immunity (Days)")
    add_panel_label(axes[2], 'C. Immune Onset Delay')

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig3_path = "figures/fig3_paired.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

if __name__ == "__main__":
    generate_fig3()
