import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from paths import result_path, figure_path

def plot_psa():
    results_file = result_path("targeted_psa_results.csv")
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return

    df = pd.read_csv(results_file)
    
    # Calculate averted cases (Base - Strategy)
    df["ring_averted"] = df["base_cases_abs"] - df["ring_cases_abs"]
    df["comm_averted"] = df["base_cases_abs"] - df["comm_cases_abs"]

    # Parameter inputs
    params = [
        "incubation_period", "infectious_period", 
        "baseline_tau", "reporting_rate", "vaccine_effect",
        "incubation_shape", "infectious_shape"
    ]
    
    # Aggregate over replicates to get the mean per parameter set
    df_agg = df.groupby("set_id").mean().reset_index()

    # 1. Tornado Plot / Sensitivity Analysis (Spearman Rank Correlation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, (target, title) in enumerate(zip(["ring_averted", "comm_averted"], ["Ring Vaccination", "Community Vaccination"])):
        correlations = []
        for p in params:
            rho, _ = spearmanr(df_agg[p], df_agg[target])
            correlations.append((p, rho))
        
        # Sort by absolute correlation
        correlations = sorted(correlations, key=lambda x: abs(x[1]))
        labels = [x[0].replace("_", " ").title() for x in correlations]
        values = [x[1] for x in correlations]
        
        colors = ['#1f77b4' if v > 0 else '#d62728' for v in values]

        axes[idx].barh(labels, values, color=colors)
        axes[idx].set_title(f"Sensitivity: {title}\n(Spearman Correlation with Averted Cases)")
        axes[idx].set_xlim(-1, 1)
        axes[idx].axvline(0, color='black', linewidth=0.8)
        axes[idx].set_xlabel("Spearman Rank Correlation Coefficient (ρ)")
    
    plt.tight_layout()
    import time
    ts = int(time.time())
    tornado_path = figure_path("polished/psa_tornado.pdf")
    os.makedirs(os.path.dirname(tornado_path), exist_ok=True)
    out_png1 = f'/Users/jasonandrews/.gemini/antigravity-ide/brain/8d115dec-1d3c-47dc-8c2a-9349afc3e8c4/psa_tornado_{ts}.png'
    plt.savefig(out_png1, dpi=300, bbox_inches='tight')
    plt.savefig(tornado_path)
    plt.savefig(tornado_path.with_suffix(".png"))
    print(f"Saved Tornado plot to {out_png1}")
    print(f"Saved Tornado plot to {tornado_path}")
    plt.close()

    # 2. Distribution Plot of Cases
    plt.figure(figsize=(8, 6))
    
    # Melt dataframe for seaborn boxplot
    melted = df_agg.melt(
        id_vars=["set_id"], 
        value_vars=["base_cases_abs", "ring_cases_abs", "comm_cases_abs"],
        var_name="Scenario", 
        value_name="Absolute Cases"
    )
    melted["Scenario"] = melted["Scenario"].replace({
        "base_cases_abs": "Baseline",
        "ring_cases_abs": "Ring Vaccination",
        "comm_cases_abs": "Community Vaccination"
    })

    sns.boxplot(x="Scenario", y="Absolute Cases", data=melted, showfliers=False, color='lightblue')
    sns.stripplot(x="Scenario", y="Absolute Cases", data=melted, color=".25", alpha=0.5, jitter=True)
    
    plt.title("Probabilistic Sensitivity Analysis:\nDistribution of Absolute Cases Across 250 Parameter Sets")
    plt.ylabel("Expected Absolute Cases (Mean of 200 Replicates)")
    
    dist_path = figure_path("polished/psa_distribution.pdf")
    plt.tight_layout()
    out_png2 = f'/Users/jasonandrews/.gemini/antigravity-ide/brain/8d115dec-1d3c-47dc-8c2a-9349afc3e8c4/psa_distribution_{ts}.png'
    plt.savefig(out_png2, dpi=300, bbox_inches='tight')
    plt.savefig(dist_path)
    plt.savefig(dist_path.with_suffix(".png"))
    print(f"Saved Distribution plot to {out_png2}")
    print(f"Saved Distribution plot to {dist_path}")
    plt.close()

if __name__ == "__main__":
    plot_psa()
