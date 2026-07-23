"""Re-render supplementary figures S2, S3, S5 without on-figure titles.

S2: calibration (A: realised Rt vs EpiNow2 input; B: cumulative simulated onsets)
S3: timing of vaccination among eventual cases (stacked bars, current-engine raw)
S5: immune-onset (A: sigmoid protection profiles; B: midpoint sensitivity bars)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures/current_review/manuscript_review_figures_20260722")
BLUE, TEAL, ORANGE, GRAY = "#2878B5", "#21918C", "#D2694E", "#667085"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#E1E4E8", lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------- S2 ----------------
z = np.load("data_and_results/review_outputs/supplementary_calibration_updated_epinow_median_20260722/supplementary_trajectory_arrays.npz")
onset, rt_real, rt_in = z["daily_onset_cases"], z["realized_rt"], z["input_rt"]
days = np.arange(onset.shape[1])
cum = np.cumsum(onset, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.6))
ax = axes[0]
ax.fill_between(days, np.percentile(rt_real, 2.5, axis=0), np.percentile(rt_real, 97.5, axis=0),
                color=TEAL, alpha=0.18, lw=0, label="Simulation 95% range")
ax.plot(days, np.median(rt_real, axis=0), color=TEAL, lw=1.8, label="Realised model Rt (median)")
ax.plot(days, np.median(rt_in, axis=0), color=GRAY, lw=1.6, ls="--", label="EpiNow2 median input")
ax.set_ylabel("Effective reproduction number", fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
style(ax)
ax.text(-0.06, 1.03, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")

ax = axes[1]
ax.fill_between(days, np.percentile(cum, 2.5, axis=0), np.percentile(cum, 97.5, axis=0),
                color=ORANGE, alpha=0.18, lw=0, label="Simulation 95% range")
ax.plot(days, np.median(cum, axis=0), color=ORANGE, lw=1.8, ls="--", label="Simulation median")
ax.set_ylabel("Cumulative onset cases", fontweight="bold")
ax.set_xlabel("Days since outbreak declaration (15 May 2026)", fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
style(ax)
ax.text(-0.06, 1.03, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Figure_S2.png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "Supplementary_Figure_S2.pdf", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("S2 done")

# ---------------- S3 ----------------
df = pd.read_csv("archive/2026-07-22_pre_github_cleanup/data_and_results_legacy/superseded_current_review/current_batch_20260722/supp_s3_delivery_raw.csv")
order = [s for s in df.strategy.unique()]
labels = {"ring2_enhanced": "Ring 2 + enhanced", "community40_base": "Community 40%"}
order = sorted(order, key=lambda s: 0 if "ring" in s else 1)
shares = {}
for s in order:
    sub = df[df.strategy == s]
    tot = sub.n_vaccinated_cases.sum()
    shares[labels.get(s, s)] = [sub.before_exposure.sum() / tot * 100,
                                sub.during_incubation.sum() / tot * 100,
                                sub.after_onset.sum() / tot * 100]

fig, ax = plt.subplots(figsize=(7.2, 2.9))
names = list(shares)
vals = np.array([shares[n] for n in names])
left = np.zeros(len(names))
for comp, color, lab in zip(range(3), (TEAL, "#D8A23C", ORANGE),
                            ("Before exposure", "During incubation", "After onset")):
    ax.barh(names, vals[:, comp], left=left, color=color, height=0.52, alpha=0.95, label=lab)
    left += vals[:, comp]
ax.set_xlabel("Vaccinated eventual cases (%)", fontweight="bold")
ax.set_xlim(0, 100)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
style(ax, grid_axis="x")
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Figure_S3.png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "Supplementary_Figure_S3.pdf", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("S3 done")

# ---------------- S5 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.25})
d = np.linspace(0, 30, 400)
k, vemax = 0.5, 45
ax = axes[0]
for d0, color, lab in ((5, ORANGE, "Fast onset (midpoint 5 d)"),
                       (10, TEAL, "Standard onset (midpoint 10 d)"),
                       (14, BLUE, "Slow onset (midpoint 14 d)")):
    ax.plot(d, vemax / (1 + np.exp(-k * (d - d0))), color=color, lw=2.2, label=lab)
ax.plot([0, 10, 10, 30], [0, 0, vemax, vemax], color=GRAY, lw=1.6, ls="--", label="Step function (day 10)")
ax.set_xlabel("Days since vaccination", fontweight="bold")
ax.set_ylabel("Protection (%)", fontweight="bold")
ax.set_ylim(0, 50)
ax.legend(frameon=False, fontsize=8.6, loc="lower right")
style(ax)
ax.text(-0.09, 1.03, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")

raw = pd.read_csv("data_and_results/review_outputs/immune_onset_paired_1000_20260723/fig4_immune_onset_paired_raw.csv")
g = raw.groupby("immune_midpoint")["mortality_reduction_pct"]
mids = sorted(raw.immune_midpoint.unique())
med = [g.get_group(m).median() for m in mids]
lo = [np.percentile(g.get_group(m), 25) for m in mids]
hi = [np.percentile(g.get_group(m), 75) for m in mids]
ax = axes[1]
ax.bar(range(len(mids)), med, color=BLUE, width=0.55, alpha=0.94,
       yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
       error_kw=dict(ecolor="#172033", elinewidth=1.1, capsize=3.5, alpha=0.8))
ax.set_xticks(range(len(mids)), [f"{int(m)}" for m in mids])
ax.set_xlabel("Immune-onset midpoint (days)", fontweight="bold")
ax.set_ylabel("Median mortality reduction (%)", fontweight="bold")
ax.set_ylim(0, max(hi) * 1.2)
style(ax)
ax.text(-0.09, 1.03, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Figure_S5.png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "Supplementary_Figure_S5.pdf", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("S5 done")
