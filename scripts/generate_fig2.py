import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import datetime
from paths import result_path

def generate_fig2():
    os.makedirs("figures", exist_ok=True)
    raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

    # Extract contour data
    contour_df = raw[raw["scenario"] == "fig2_contour"].copy()
    
    if len(contour_df) == 0:
        print("Warning: No fig2_contour data found. Run the cluster job first.")
        return
        
    contour_df["is_vax"] = contour_df["level"].apply(lambda x: x.startswith("vax_"))
    contour_df["det"] = contour_df["level"].apply(lambda x: float(x.split('_')[1]))
    contour_df["ve"] = contour_df["level"].apply(lambda x: float(x.split('_')[2]))

    df_base = contour_df[~contour_df["is_vax"]].groupby(["det", "ve"])["deaths_percent"].median().reset_index()
    df_vax = contour_df[contour_df["is_vax"]].groupby(["det", "ve"])["deaths_percent"].median().reset_index()

    merged = pd.merge(df_base, df_vax, on=["det", "ve"], suffixes=("_base", "_vax"))
    
    # Paired Delta: Percentage Deaths Averted
    merged["deaths_averted"] = np.where(merged["deaths_percent_base"] > 0,
                                        (merged["deaths_percent_base"] - merged["deaths_percent_vax"]) / merged["deaths_percent_base"] * 100.0,
                                        0.0)

    # Prepare for contour plot
    D = merged["det"].values
    V = merged["ve"].values
    Z = merged["deaths_averted"].values

    Di, Vi = np.mgrid[0.4:0.9:100j, 0.2:0.9:100j]
    Zi = griddata((D, V), Z, (Di, Vi), method='cubic')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    contour = ax.contourf(Vi, Di, Zi, levels=20, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Paired Deaths Averted (%)")

    # Mark base case (VE=0.45, Det=0.7)
    ax.scatter([0.45], [0.70], color="#2ecc71", marker="*", s=250, edgecolor="black", label="Base Case")

    ax.set_xlabel("Vaccine Efficacy")
    ax.set_ylabel("Index Case Detection Rate")
    ax.set_title("A. Impact of Surveillance and Vaccine Efficacy")
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig2_path = f"figures/fig2_paired.png"
    plt.savefig(fig2_path)
    plt.close()

if __name__ == "__main__":
    generate_fig2()
