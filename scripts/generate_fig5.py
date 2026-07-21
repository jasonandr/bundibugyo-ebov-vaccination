import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paths import result_path

def generate_fig5():
    os.makedirs("figures", exist_ok=True)
    raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

    # The base case for the Tornado plot is the Enhanced Vax scenario
    base_case_df = raw[(raw["scenario"] == "analysis_1_reactive_ring") & (raw["level"] == "vax_enh_ops")]
    if len(base_case_df) == 0:
        print("Warning: Base case missing.")
        return
        
    base_deaths = base_case_df["deaths_percent"].median()

    ranges = [
        ("Case Detection\n(40% to 90%)", "fig5_tornado_det", "vax_det_0.4", "vax_det_0.9"),
        ("Tracing Coverage\n(40% to 90%)", "fig5_tornado_trace", "vax_trace_0.4", "vax_trace_0.9"),
        ("Vaccine Efficacy\n(50% to 100%)", "fig5_tornado_eff", "vax_eff_0.5", "vax_eff_1.0"),
        ("Vaccine Uptake\n(60% to 100%)", "fig5_tornado_uptake", "vax_uptake_0.6", "vax_uptake_1.0"),
        ("Tracing Delay\n(7 days to 1 day)", "fig5_tornado_delay", "vax_delay_7.0", "vax_delay_1.0"),
        ("Immune Onset\n(14 days to 5 days)", "fig5_tornado_immune", "vax_immune_14.0", "vax_immune_5.0")
    ]

    results = []
    for label, sc, lvl_worst, lvl_best in ranges:
        worst_df = raw[(raw["scenario"] == sc) & (raw["level"] == lvl_worst)]
        best_df = raw[(raw["scenario"] == sc) & (raw["level"] == lvl_best)]
        
        if len(worst_df) > 0 and len(best_df) > 0:
            worst_deaths = worst_df["deaths_percent"].median()
            best_deaths = best_df["deaths_percent"].median()
            
            # The absolute reduction in mortality % relative to the base case
            worst_delta = base_deaths - worst_deaths 
            best_delta = base_deaths - best_deaths
            
            results.append({
                'Variable': label,
                'Worst_Value': worst_delta,
                'Best_Value': best_delta,
                'Impact': abs(worst_delta - best_delta)
            })

    if not results:
        print("No tornado data found.")
        return
        
    df_plot = pd.DataFrame(results)
    df_plot = df_plot.sort_values('Impact', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Positive delta means MORE deaths (worse), Negative delta means FEWER deaths (better)
    # We will plot the bars diverging from 0
    ax.barh(df_plot['Variable'], df_plot['Worst_Value'], color='#e74c3c', label='Worst-case limit', height=0.5, zorder=3)
    ax.barh(df_plot['Variable'], df_plot['Best_Value'], color='#2ecc71', label='Best-case limit', height=0.5, zorder=3)

    # X ticks will now directly show the relative percentage
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    
    ax.axvline(0, color='black', linestyle='--', linewidth=2, zorder=4, label='Base Case (0% Reduction)')
    ax.set_xlabel("Reduction in mortality compared with base case (%)")
    ax.set_title(f"Sensitivity of Mortality to Parameters (Base Mortality = {base_deaths:.2f}%)")
    
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

    plt.tight_layout()
    fig5_path = "figures/fig5_paired.png"
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved Figure 5 to {fig5_path}")

if __name__ == "__main__":
    generate_fig5()
