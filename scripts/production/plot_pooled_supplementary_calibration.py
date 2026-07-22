"""Render S1/S2 review figures from already generated pooled-model trajectories.

This intentionally runs in a fresh Python process after the C++ trajectory
generator has completed.  It never alters the trajectory arrays.
"""
import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


REPO = Path(__file__).resolve().parents[2]


def observed_cases():
    source = REPO / "BDBV2026-Data" / "build" / "long" / "insp_sitrep__national_cumulative_confirmed_cases.csv"
    values = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3 and row[0] == "DRC":
                values[datetime.fromisoformat(row[1])] = float(row[2])
    override = REPO / "data_and_results" / "sitrep_2026_07_05_override.csv"
    if override.exists():
        with override.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["country"] == "DRC":
                    values[datetime.fromisoformat(row["date"])] = float(row["confirmed_cases"])
    dates = sorted(values)
    case_values = np.maximum.accumulate(np.array([values[date] for date in dates], dtype=float))
    daily_dates = [dates[0] + timedelta(days=i) for i in range((dates[-1] - dates[0]).days + 1)]
    daily_values = np.interp([date.timestamp() for date in daily_dates],
                             [date.timestamp() for date in dates], case_values)
    return daily_dates, daily_values


def epinow_summary(days):
    """Read the authoritative updated EpiNow2 summary directly, not a legacy NPY."""
    source = REPO / "results" / "epinow_rt.csv"
    values = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["type"].startswith("estimate"):
                values[datetime.fromisoformat(row["date"])] = (
                    float(row["lower_90"]), float(row["median"]), float(row["upper_90"])
                )
    dates = sorted(values)[:days]
    if len(dates) < days:
        dates.extend([dates[-1] + timedelta(days=i) for i in range(1, days - len(dates) + 1)])
    low, median, high = [], [], []
    last = values[sorted(values)[-1]]
    for date in dates:
        value = values.get(date, last)
        low.append(value[0]); median.append(value[1]); high.append(value[2])
    return dates, np.asarray(low), np.asarray(median), np.asarray(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-s1", action="store_true", help="Resume after a successfully rendered S1")
    parser.add_argument("--s2-only", action="store_true", help="Render only S2")
    parser.add_argument("--omit-reported-cases", action="store_true",
                        help="Do not plot DRC-wide notifications against the 100,000-person model network")
    args = parser.parse_args()
    s1 = args.output_dir / "Supplementary_Figure_S1_pooled_review.png"
    s2 = args.output_dir / "Supplementary_Figure_S2_pooled_review.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (s1.exists() and not (args.skip_s1 or args.s2_only)) or s2.exists():
        raise FileExistsError("Refusing to overwrite an existing review figure")
    with np.load(args.trajectory_file) as values:
        incidence = values["daily_onset_cases"]
        realized_rt = values["realized_rt"]
        input_rt = values["input_rt"]
    print("Trajectory arrays loaded.", flush=True)
    n, days = incidence.shape
    dates, input_low, input_mid, input_high = epinow_summary(days)
    if not args.omit_reported_cases:
        observed_dates, observed_values = observed_cases()
        observed_lookup = dict(zip(observed_dates, observed_values))
        observed_series = np.array([observed_lookup.get(date, np.nan) for date in dates])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    print("Drawing S1.", flush=True)
    if not (args.skip_s1 or args.s2_only):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for row in realized_rt:
            ax.plot(dates, row, color="#c0392b", alpha=0.10, lw=0.8)
        print("S1 trajectories drawn.", flush=True)
        ax.fill_between(dates, input_low, input_high, color="#5dade2", alpha=0.30, label="EpiNow2 90% CrI")
        ax.plot(dates, input_mid, color="#2874a6", lw=2.2, label=r"EpiNow2 median $R_t$")
        ax.plot(dates, np.nanmedian(realized_rt, axis=0), color="#c0392b", lw=2.8, ls="--",
                label=rf"Simulated realised $R_t$ (median; n={n})")
        ax.axhline(1, color="#566573", ls=":", lw=1.5)
        ax.set(title=r"Supplementary Figure S1: EpiNow2 input and simulated realised $R_t$",
               ylabel=r"Effective reproduction number ($R_t$)", xlabel="Date (2026)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#d5d8dc", ls="--", alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(s1, dpi=300, facecolor="white")
        plt.close(fig)
        print("S1 saved.", flush=True)

    cumulative = np.cumsum(incidence, axis=1)
    low, mid, high = np.quantile(cumulative, [0.025, 0.50, 0.975], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in cumulative:
        ax.plot(dates, row, color="#c0392b", alpha=0.06, lw=0.7)
    ax.fill_between(dates, low, high, color="#f5b7b1", alpha=0.55, label="Simulation 95% range")
    ax.plot(dates, mid, color="#c0392b", lw=2.8, ls="--", label="Simulation median")
    if not args.omit_reported_cases:
        observed_mask = np.isfinite(observed_series)
        observed_dates_for_plot = np.asarray(dates, dtype=object)[observed_mask]
        ax.plot(observed_dates_for_plot, observed_series[observed_mask], color="#34495e", lw=2.2,
                label="Reported cumulative confirmed cases")
    ax.set(title="Supplementary Figure S2: Cumulative simulated cases",
           ylabel="Cumulative cases", xlabel="Date (2026)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d5d8dc", ls="--", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(s2, dpi=300, facecolor="white")
    plt.close(fig)
    (args.output_dir / "figure_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "trajectory_file": str(args.trajectory_file),
        "figures": [s2.name] if args.s2_only else [s1.name, s2.name],
        "note": "Review-only figures; no submission figure was overwritten.",
        "reported_cases_plotted": not args.omit_reported_cases,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
