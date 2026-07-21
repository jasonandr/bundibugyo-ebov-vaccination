import time

import matplotlib.pyplot as plt
import pandas as pd

from paths import figure_path


def plot_empirical_data(max_date=None):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df = pd.read_csv(
        "BDBV2026-Data/data/insp_sitrep/processed/insp_sitrep__new_confirmed_cases__daily.csv"
    )
    df["new_confirmed_cases"] = pd.to_numeric(
        df["new_confirmed_cases"], errors="coerce"
    ).fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.sort_values("date").groupby("date")["new_confirmed_cases"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily["date"], daily["new_confirmed_cases"], color="#34495E", alpha=0.8)
    ax.set_ylabel("Daily confirmed cases", fontsize=12)
    ax.set_title("Empirical outbreak data (BDBV 2026)", loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if max_date is not None:
        ax.set_xlim(daily["date"].min() - pd.Timedelta(days=1), pd.to_datetime(max_date))

    fig.autofmt_xdate()
    fig.tight_layout()
    out = figure_path(f"empirical_data_only_{int(time.time())}.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved empirical incidence plot to {out}")


if __name__ == "__main__":
    plot_empirical_data()
