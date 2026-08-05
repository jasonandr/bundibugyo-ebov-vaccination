"""Render Figure 1: model schematic, clustered contact network, and calibration.

Panel A: SEIR compartment schema with the intervention timeline.
Panel B: index case with radius-1 and radius-2 neighbourhoods on an
         illustrative three-layer clustered network (generate_network_clustered).
Panel C: observed daily incidence with the EpiNow2 Rt estimate.

Outputs: figures/final/Figure_1.png (300 dpi) and Figure_1.pdf (vector).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd

from current_outbreak_data import cumulative_confirmed_cases, cumulative_confirmed_deaths
from ebola_stochastic_ring import generate_network_clustered

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/final/Figure_1"
RT_CSV = ROOT / "scripts/results/epinow_rt.csv"


def panel_schematic(ax):
    ax.axis("off")
    ax.set_title("A", loc="left", fontsize=24, fontweight="bold")

    boxes = [
        ("Susceptible\n(S)", "#4A90E2", 0.1),
        ("Exposed\n(E)", "#F39C12", 0.35),
        ("Infectious\n(I)", "#E74C3C", 0.6),
        ("Removed\n(R)", "#7F8C8D", 0.85),
    ]
    box_w, box_h = 0.18, 0.15
    for text, color, x in boxes:
        rect = plt.Rectangle((x - box_w / 2, 0.6), box_w, box_h, color=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 0.6 + box_h / 2, text, color="white", ha="center", va="center",
                fontweight="bold", fontsize=12, zorder=3)

    for i in range(3):
        x_start = boxes[i][2] + box_w / 2 + 0.01
        x_end = boxes[i + 1][2] - box_w / 2 - 0.01
        ax.annotate("", xy=(x_end, 0.6 + box_h / 2), xytext=(x_start, 0.6 + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2))

    ax.text((boxes[0][2] + boxes[1][2]) / 2, 0.6 + box_h / 2 + 0.03, r"$\lambda$", ha="center", fontsize=14)
    ax.text((boxes[1][2] + boxes[2][2]) / 2, 0.6 + box_h / 2 + 0.03, r"$\sigma$", ha="center", fontsize=14)
    ax.text((boxes[2][2] + boxes[3][2]) / 2, 0.6 + box_h / 2 + 0.03, r"$\gamma$", ha="center", fontsize=14)

    y_timeline = 0.3
    events = [(0.1, "Exposure"), (0.35, "Onset"), (0.5, "Detection"), (0.7, "Vaccinate"), (0.9, "Immunity")]
    ax.plot([0.1, 0.9], [y_timeline, y_timeline], color="#2C3E50", lw=2)
    for x, label in events:
        ax.plot([x, x], [y_timeline - 0.02, y_timeline + 0.02], color="#2C3E50", lw=2)
        ax.text(x, y_timeline - 0.04, label, ha="center", va="top", fontweight="bold",
                fontsize=12, color="#2C3E50")

    ax.plot([0.1, 0.1], [y_timeline, 0.6], color="#BDC3C7", ls=":", lw=2)
    ax.plot([0.35, 0.35], [y_timeline, 0.6], color="#BDC3C7", ls=":", lw=2)

    ax.annotate("", xy=(0.35, y_timeline + 0.04), xytext=(0.1, y_timeline + 0.04),
                arrowprops=dict(arrowstyle="<->", color="#7F8C8D"))
    ax.text(0.225, y_timeline + 0.06, "Incubation\nPeriod", ha="center", va="bottom", fontsize=10, color="#7F8C8D")

    ax.annotate("", xy=(0.5, y_timeline + 0.04), xytext=(0.35, y_timeline + 0.04),
                arrowprops=dict(arrowstyle="<->", color="#7F8C8D"))
    ax.text(0.425, y_timeline + 0.06, "Reporting\nDelay", ha="center", va="bottom", fontsize=10, color="#7F8C8D")

    ax.annotate("", xy=(0.7, y_timeline + 0.04), xytext=(0.5, y_timeline + 0.04),
                arrowprops=dict(arrowstyle="<->", color="#F39C12"))
    ax.text(0.6, y_timeline + 0.06, "Tracing\nDelay", ha="center", va="bottom", fontsize=10,
            color="#F39C12", fontweight="bold")

    ax.annotate("", xy=(0.9, y_timeline + 0.04), xytext=(0.7, y_timeline + 0.04),
                arrowprops=dict(arrowstyle="<->", color="#27AE60"))
    ax.text(0.8, y_timeline + 0.06, "Immune\nDelay", ha="center", va="bottom", fontsize=10,
            color="#27AE60", fontweight="bold")

    ax.text(0.3, y_timeline - 0.12, "Index Cases", ha="center", fontsize=12, fontweight="bold", color="#E74C3C")
    ax.plot([0.1, 0.5], [y_timeline - 0.10, y_timeline - 0.10], color="#E74C3C", lw=2)

    ax.text(0.8, y_timeline - 0.12, "Contacts", ha="center", fontsize=12, fontweight="bold", color="#F39C12")
    ax.plot([0.7, 0.9], [y_timeline - 0.10, y_timeline - 0.10], color="#F39C12", lw=2)


def panel_network(ax):
    ax.set_title("B", loc="left", fontsize=24, fontweight="bold")
    ax.axis("off")

    # Illustrative small draw of the three-layer clustered topology used in
    # the simulations: household cliques packed into local community clusters
    # (wired through overlapping small groups) plus overdispersed
    # inter-cluster stubs. Nodes are coloured by network distance from the
    # index case.
    G = generate_network_clustered(N=300, cluster_mean=40.0, cluster_sd=10.0,
                                   inner_lambda=0.5, inner_size_mean=5.0, inner_size_sd=2.0,
                                   stub_mean=1.5, stub_var=30.0, seed=1)
    degrees = dict(G.degree())
    index_case = max(degrees, key=degrees.get)

    lengths = nx.single_source_shortest_path_length(G, index_case, cutoff=3)
    rad1 = [n for n, d in lengths.items() if d == 1]
    rad2 = [n for n, d in lengths.items() if d == 2]
    rad3 = [n for n, d in lengths.items() if d == 3]
    far = [n for n in G.nodes if n not in lengths]

    pos = nx.spring_layout(G, seed=42, k=0.35)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, nodelist=far, node_color="#DDDDDD", node_size=20, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=rad3, node_color="#ecf0f1", node_size=30, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=rad2, node_color="#f1c40f", node_size=50, ax=ax,
                           label="Radius 2 (Contacts of Contacts)")
    nx.draw_networkx_nodes(G, pos, nodelist=rad1, node_color="#e67e22", node_size=100, ax=ax,
                           label="Radius 1 (Direct Contacts)")
    nx.draw_networkx_nodes(G, pos, nodelist=[index_case], node_color="#c0392b", node_size=300, ax=ax,
                           label="Index Case")
    ax.legend(loc="lower left", frameon=False, fontsize=12)


def load_incidence():
    cases_df = cumulative_confirmed_cases().sort_values("Date")
    deaths_df = cumulative_confirmed_deaths().sort_values("Date")

    cases_raw = cases_df["Cases"].values.copy()
    for i in range(len(cases_raw) - 2, -1, -1):
        if cases_raw[i] > cases_raw[i + 1]:
            cases_raw[i] = cases_raw[i + 1]
    cases_df["Cases"] = cases_raw

    date_range = pd.date_range(start=cases_df["Date"].min(), end=cases_df["Date"].max(), freq="D")
    df_daily = pd.merge(pd.DataFrame({"Date": date_range}), cases_df, on="Date", how="left")
    df_daily["Cases"] = df_daily["Cases"].interpolate(method="linear")
    cases_inc = np.diff(df_daily["Cases"].values, prepend=df_daily["Cases"].values[0])

    deaths_raw = deaths_df["Deaths"].values.copy()
    for i in range(len(deaths_raw) - 2, -1, -1):
        if deaths_raw[i] > deaths_raw[i + 1]:
            deaths_raw[i] = deaths_raw[i + 1]
    deaths_df["Deaths"] = deaths_raw
    df_daily_deaths = pd.merge(pd.DataFrame({"Date": date_range}), deaths_df, on="Date", how="left")
    df_daily_deaths["Deaths"] = df_daily_deaths["Deaths"].interpolate(method="linear").fillna(0)
    deaths_inc = np.diff(df_daily_deaths["Deaths"].values, prepend=df_daily_deaths["Deaths"].values[0])
    return df_daily, cases_inc, df_daily_deaths, deaths_inc


def panel_calibration(ax):
    ax.set_title("C", loc="left", fontsize=24, fontweight="bold")
    df_daily, cases_inc, df_daily_deaths, deaths_inc = load_incidence()

    ax.bar(df_daily["Date"], cases_inc, color="#5DADE2", width=1.0, alpha=0.6,
           edgecolor="black", linewidth=0.5, label="Confirmed Cases")
    cases_inc_ma = pd.Series(cases_inc).rolling(window=7, min_periods=1).mean().values
    ax.plot(df_daily["Date"], cases_inc_ma, color="#2980B9", lw=2.5, label="Cases (7-day MA)")
    ax.bar(df_daily_deaths["Date"], -deaths_inc, color="#5D6D7E", width=1.0,
           edgecolor="black", linewidth=0.5, label="Confirmed Deaths")
    ax.set_ylabel("Daily Incidence (Cases $\\uparrow$, Deaths $\\downarrow$)", fontsize=14)
    ax.axhline(0, color="black", lw=1.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.set_xlabel("Date (2026)", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax_twin = ax.twinx()
    rt_df = pd.read_csv(RT_CSV)
    rt_df["date"] = pd.to_datetime(rt_df["date"])
    rt_len = min(len(rt_df), len(df_daily["Date"]))
    ax_twin.fill_between(rt_df["date"][:rt_len], rt_df["lower_90"][:rt_len],
                         rt_df["upper_90"][:rt_len], color="#C0392B", alpha=0.2, label="90% CrI")
    ax_twin.plot(rt_df["date"][:rt_len], rt_df["median"][:rt_len], color="#C0392B",
                 lw=3, label="Estimated $R_t$")
    ax_twin.axhline(1.0, color="gray", linestyle="--", alpha=0.5, lw=2)
    ax_twin.set_ylabel("Effective Reproduction Number ($R_t$)", fontsize=14, color="#C0392B")
    ax_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax_twin.set_ylim(0, max(6.0, np.nanmax(rt_df["upper_90"]) * 1.1))

    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax_twin.spines["top"].set_visible(False)
    max_cases = max(cases_inc.max(), deaths_inc.max())
    ax.set_ylim(-max_cases * 1.1, max_cases * 1.1)
    return df_daily["Date"].max()


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(24, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    panel_schematic(fig.add_subplot(gs[0]))
    panel_network(fig.add_subplot(gs[1]))
    last_date = panel_calibration(fig.add_subplot(gs[2]))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    print(f"Wrote {OUT.with_suffix('.png')} / {OUT.with_suffix('.pdf')}; data through {last_date.date()}")


if __name__ == "__main__":
    main()
