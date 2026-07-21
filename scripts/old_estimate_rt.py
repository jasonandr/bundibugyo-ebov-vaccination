import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma

from paths import figure_path, result_path


def estimate_rt():
    """Legacy renewal-equation Rt estimator retained for comparison."""
    df = pd.read_csv(
        "BDBV2026-Data/build/long/insp_sitrep__national_cumulative_confirmed_cases.csv",
        header=None,
        names=["Country", "Date", "Cases"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    cases_raw = df["Cases"].values.copy()
    for i in range(len(cases_raw) - 2, -1, -1):
        if cases_raw[i] > cases_raw[i + 1]:
            cases_raw[i] = cases_raw[i + 1]
    df["Cases"] = cases_raw

    date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
    df_daily = pd.DataFrame({"Date": date_range}).merge(df, on="Date", how="left")
    df_daily["Cases"] = df_daily["Cases"].interpolate(method="linear")

    cases_inc = np.diff(df_daily["Cases"].values, prepend=df_daily["Cases"].iloc[0])
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g) ** 2
    scale = (std_g**2) / mean_g
    weights = gamma.pdf(np.arange(1, len(cases_inc) + 1), a=shape, scale=scale)
    weights = weights / weights.sum()

    infectiousness = np.zeros(len(cases_inc))
    for t in range(1, len(cases_inc)):
        for lag in range(1, t + 1):
            infectiousness[t] += cases_inc[t - lag] * weights[lag - 1]

    rt = np.zeros(len(cases_inc))
    mask = infectiousness > 0
    rt[mask] = cases_inc[mask] / infectiousness[mask]
    rt_smooth = pd.Series(rt).rolling(window=7, min_periods=1, center=True).mean().values

    params_file = result_path("fitted_parameters_legacy.json")
    with open(params_file, "w") as f:
        json.dump({"Rt_array": rt_smooth.tolist(), "Rend": float(rt_smooth[-1])}, f, indent=4)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily["Date"], rt_smooth, color="red", linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Effective reproduction number")
    ax.set_title("Legacy renewal-equation Rt estimate")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = figure_path(f"legacy_rt_estimation_{int(time.time())}.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved legacy Rt estimate to {out}")


if __name__ == "__main__":
    estimate_rt()
