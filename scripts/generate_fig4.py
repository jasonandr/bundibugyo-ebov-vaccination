import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import datetime
from paths import result_path

def generate_fig4():
    os.makedirs("figures", exist_ok=True)
    raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

    # Extract contour data
    contour_df = raw[raw["scenario"] == "fig4_heatmap"].copy()
    
    if len(contour_df) == 0:
        print("Warning: No fig4_heatmap data found. Run the cluster job first.")
        return
        
    contour_df["is_vax"] = contour_df["level"].apply(lambda x: x.startswith("vax_"))
    contour_df["det"] = contour_df["level"].apply(lambda x: float(x.split('_')[1]))
    contour_df["tracing"] = contour_df["level"].apply(lambda x: float(x.split('_')[2]))

    df_base = contour_df[~contour_df["is_vax"]].groupby(["det", "tracing"])["deaths_percent"].median().reset_index()
    df_vax = contour_df[contour_df["is_vax"]].groupby(["det", "tracing"])["deaths_percent"].median().reset_index()

    merged = pd.merge(df_base, df_vax, on=["det", "tracing"], suffixes=("_base", "_vax"))
    
    # Paired Delta: Percentage Deaths Averted
    merged["deaths_averted"] = np.where(merged["deaths_percent_base"] > 0,
                                        (merged["deaths_percent_base"] - merged["deaths_percent_vax"]) / merged["deaths_percent_base"] * 100.0,
                                        0.0)

    # Prepare for contour plot
    D = merged["det"].values
    T = merged["tracing"].values
    Z = merged["deaths_averted"].values

    Di, Ti = np.mgrid[0.3:0.9:100j, 0.3:0.9:100j]
    Zi = griddata((D, T), Z, (Di, Ti), method='cubic')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    contour = ax.contourf(Ti, Di, Zi, levels=20, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Paired Deaths Averted (%)")

    # Mark base case (Det=0.7, Tracing=0.8)
    ax.scatter([0.80], [0.70], color="#2ecc71", marker="*", s=250, edgecolor="black", label="Base Case")

    ax.set_xlabel("Contact Tracing Coverage")
    ax.set_ylabel("Index Case Detection Rate")
    ax.set_title("Operational Contour Pairs: Detection vs. Tracing")
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig4_path = f"figures/fig4_paired.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved Figure 4 to {fig4_path}")

if __name__ == "__main__":
    generate_fig4()
