"""Render Supplementary Figure S4: independent VE contour (smoothed)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SUMMARY = REPO / "data_and_results/review_outputs/s4_independent_ve_fine_20260723/s4_grid_summary.csv"
import os
OUT = REPO / os.environ.get("S4_OUT", "figures/current_review/manuscript_review_figures_20260722/Supplementary_Figure_S4_Independent_VE.png")

PROBE_CELLS = [(0.15, 0.0), (0.15, 0.9), (0.0, 0.0), (0.0, 0.9), (0.45, 0.45)]


def main():
    with SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grid_rows = [r for r in rows if r["cell"] != "base_no_vax"]
    base = next(r for r in rows if r["cell"] == "base_no_vax")
    ve_i = np.array([float(r["ve_i"]) for r in grid_rows])
    ve_m = np.array([float(r["ve_m"]) for r in grid_rows])
    reduction = np.array([float(r["median_mortality_reduction_pct"]) for r in grid_rows])
    median_deaths = {(float(r["ve_i"]), float(r["ve_m"])): float(r["median_deaths"])
                     for r in grid_rows}

    fine = np.linspace(0.0, 0.90, 400)
    xi, yi = np.meshgrid(fine, fine)
    zi = griddata(np.column_stack([ve_i, ve_m]), reduction, (xi, yi), method="cubic")
    zi = gaussian_filter(np.nan_to_num(zi, nan=0.0), sigma=1.0)
    zi = np.clip(zi, 0.0, 100.0)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    cf = ax.contourf(xi * 100, yi * 100, zi, levels=np.linspace(0, 100, 21),
                     cmap="YlGnBu", vmin=0, vmax=100, extend="neither")
    cs = ax.contour(xi * 100, yi * 100, zi, levels=[20, 40, 60, 80],
                    colors="black", linewidths=0.8)
    labels = ax.clabel(cs, inline=True, fontsize=9, fmt="%d%%")
    for text in labels:
        text.set_rotation(0)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Median mortality reduction (%)")
    ax.set_xlabel("Vaccine effectiveness against infection (%)")
    ax.set_ylabel("Vaccine effectiveness against mortality (%)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print(f"Wrote {OUT}")

    print(f"Base arm (no vax) median deaths: {float(base['median_deaths']):.0f}")
    for cell in PROBE_CELLS:
        print(f"median deaths at ve_i={cell[0]:.2f}, ve_m={cell[1]:.2f}: "
              f"{median_deaths[cell]:.0f}")


if __name__ == "__main__":
    main()
