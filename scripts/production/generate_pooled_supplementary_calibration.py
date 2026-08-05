"""Generate review copies of Supplementary Figures S1 and S2.

The figures are deliberately written to a new output directory.  They use the
production daily onset-cohort allocator, the fixed 5.2/30/160 network cache,
posterior EpiNow2 Rt trajectories, gamma-distributed natural-history times,
and the 15 infectious plus 15 exposed initial state.  They do not overwrite
any submission-ready figure.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from ebola_stochastic_ring import simulate_ring_vaccination
from network_cache import load_cached_network


REPO = Path(__file__).resolve().parents[2]
HORIZON = 90


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lhs_draws(n_draws, n_rt_samples, seed):
    """Same nine uncertain quantities as the corrected Figure 2 PSA."""
    unit = qmc.LatinHypercube(d=9, seed=seed).random(n=n_draws)
    return [
        {
            "rt_posterior_index": min(n_rt_samples - 1, int(unit[i, 1] * n_rt_samples)),
            "incubation_mean": float(norm.ppf(0.05 + 0.90 * unit[i, 2], loc=8.5, scale=1.0)),
            "infectious_mean": float(norm.ppf(0.05 + 0.90 * unit[i, 3], loc=6.0, scale=0.8)),
            "detection_delay": float(qmc.scale(unit[i:i + 1, 4:5], [3.0], [5.0])[0, 0]),
            "incubation_shape": float(qmc.scale(unit[i:i + 1, 7:8], [1.0], [3.0])[0, 0]),
            "infectious_shape": float(qmc.scale(unit[i:i + 1, 8:9], [1.0], [3.0])[0, 0]),
        }
        for i in range(n_draws)
    ]


def rolling_ratio(numerator, denominator, window=14):
    num = pd.Series(numerator).rolling(window, min_periods=1, center=True).sum().to_numpy()
    den = pd.Series(denominator).rolling(window, min_periods=1, center=True).sum().to_numpy()
    out = np.full(len(num), np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def observed_cases():
    source = REPO / "BDBV2026-Data" / "build" / "long" / "insp_sitrep__national_cumulative_confirmed_cases.csv"
    df = pd.read_csv(source, header=None, names=["Country", "Date", "Cases"])
    df = df.loc[df["Country"].astype(str).str.replace("\ufeff", "", regex=False) == "DRC"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    # The current production data pipeline uses this reviewed 5 July update.
    override = REPO / "data_and_results" / "sitrep_2026_07_05_override.csv"
    if override.exists():
        revised = pd.read_csv(override)
        revised = revised.loc[revised["country"] == "DRC", ["date", "confirmed_cases"]].copy()
        revised["Date"] = pd.to_datetime(revised.pop("date"))
        revised["Cases"] = pd.to_numeric(revised.pop("confirmed_cases"), errors="coerce")
        df = pd.concat([df[["Date", "Cases"]], revised[["Date", "Cases"]]], ignore_index=True)
        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    df = df.sort_values("Date").copy()
    values = df["Cases"].to_numpy(dtype=float)
    # Preserve the last revised value at each date and enforce a cumulative series.
    values = np.maximum.accumulate(values)
    daily = pd.DataFrame({"Date": pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")})
    daily = daily.merge(pd.DataFrame({"Date": df["Date"], "Cases": values}), on="Date", how="left")
    daily["Cases"] = daily["Cases"].interpolate().ffill().bfill()
    return daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026072202)
    parser.add_argument("--rt-source", choices=("fitted-median", "posterior"), default="fitted-median")
    parser.add_argument("--trajectory-only", action="store_true", help="Write arrays only; render in a fresh plotting process.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.trajectories < 10:
        raise ValueError("At least 10 trajectories are required")

    fitted_path = REPO / "data_and_results" / "fitted_parameters.json"
    posterior_path = REPO / "data_and_results" / "rt_posterior_samples.npy"
    fitted = json.loads(fitted_path.read_text())
    if args.rt_source == "fitted-median":
        posterior = np.asarray([fitted["Rt_array"]], dtype=float)
    else:
        posterior = np.load(posterior_path)
        if posterior.ndim != 2:
            raise ValueError("Rt posterior samples must be a two-dimensional array")
    graph = load_cached_network(args.network_cache)
    draws = lhs_draws(args.trajectories, posterior.shape[0], args.seed)

    incidence, realized_rt, input_rt = [], [], []
    for i, draw in enumerate(draws):
        rt = posterior[draw["rt_posterior_index"]].astype(float).tolist()
        rt.extend([rt[-1]] * max(0, HORIZON + 1 - len(rt)))
        result = simulate_ring_vaccination(
            graph, rt_array=rt, baseline_tau=float(fitted.get("baseline_tau", 0.25)),
            incubation_period=draw["incubation_mean"], infectious_period=draw["infectious_mean"],
            ring_radius=2, efficacy=0.0, uptake=0.8,
            reporting_rate=[0.30] * (HORIZON + 1), tracing_coverage=[0.30] * (HORIZON + 1),
            vaccine_acceptability=1.0, detection_delay=draw["detection_delay"], tracing_delay=2.0,
            max_daily_traces=100, max_vaccines=0, base_CFR=float(fitted["base_CFR"]),
            initial_infected=15, initial_exposed=15, max_sim_time=HORIZON,
            return_time_series=True, seed=args.seed + i, engine="cpp",
            incubation_shape=draw["incubation_shape"], infectious_shape=draw["infectious_shape"],
        )
        incidence.append(np.asarray(result["daily_incidence"], dtype=float)[: HORIZON + 1])
        realized_rt.append(rolling_ratio(result["true_rt_numerator"][: HORIZON + 1],
                                         result["true_rt_denominator"][: HORIZON + 1]))
        input_rt.append(np.asarray(rt[: HORIZON + 1], dtype=float))

    incidence = np.asarray(incidence)
    realized_rt = np.asarray(realized_rt)
    input_rt = np.asarray(input_rt)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "supplementary_trajectory_arrays.npz",
                        daily_onset_cases=incidence, realized_rt=realized_rt, input_rt=input_rt)

    if args.trajectory_only:
        (args.output_dir / "manifest.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": "review-only S1/S2 trajectories",
            "trajectories": args.trajectories, "horizon_days": HORIZON, "rt_source": args.rt_source,
            "initial_state": {"infectious": 15, "exposed": 15},
            "network_cache": str(args.network_cache), "allocator": "daily pooled onset cohort",
            "waiting_times": "Gamma; PSA-sampled means and shapes",
            "cpp_sha256": sha256(REPO / "scripts/production/ebola_stochastic_ring_cpp.cpp"),
            "wrapper_sha256": sha256(REPO / "scripts/production/ebola_stochastic_ring.py"),
            "fitted_parameters_sha256": sha256(fitted_path),
            "rt_posterior_sha256": sha256(posterior_path) if args.rt_source == "posterior" else None,
        }, indent=2) + "\n")
        return

    print("Trajectory arrays written; drawing supplementary figures.", flush=True)
    observed = observed_cases()
    print("Observed case series loaded.", flush=True)
    dates = pd.date_range(observed["Date"].min(), periods=HORIZON + 1, freq="D")
    observed = observed.set_index("Date").reindex(dates)
    observed_cases_series = observed["Cases"].to_numpy(dtype=float)
    posterior_forcing = input_rt[:, :len(dates)]

    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    print("Starting S1.", flush=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in realized_rt:
        ax.plot(dates, row, color="#c0392b", alpha=0.10, lw=0.8)
    print("S1 trajectories drawn.", flush=True)
    input_low, input_mid, input_high = np.nanquantile(posterior_forcing, [0.05, 0.50, 0.95], axis=0)
    realized_mid = np.nanmedian(realized_rt, axis=0)
    ax.fill_between(dates, input_low, input_high, color="#5dade2", alpha=0.30, label="EpiNow2 90% CrI")
    ax.plot(dates, input_mid, color="#2874a6", lw=2.2, label=r"EpiNow2 median $R_t$")
    ax.plot(dates, realized_mid, color="#c0392b", lw=2.8, ls="--",
            label=rf"Simulated realised $R_t$ (median; n={args.trajectories})")
    ax.axhline(1, color="#566573", ls=":", lw=1.5)
    ax.set(title=r"Supplementary Figure S1: EpiNow2 input and simulated realised $R_t$",
           ylabel=r"Effective reproduction number ($R_t$)", xlabel="Date (2026)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d5d8dc", ls="--", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.output_dir / "Supplementary_Figure_S1_pooled_review.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("S1 written.", flush=True)

    cumulative = np.cumsum(incidence, axis=1)
    low, mid, high = np.quantile(cumulative, [0.025, 0.50, 0.975], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in cumulative:
        ax.plot(dates, row, color="#c0392b", alpha=0.06, lw=0.7)
    ax.fill_between(dates, low, high, color="#f5b7b1", alpha=0.55, label="Simulation 95% range")
    ax.plot(dates, mid, color="#c0392b", lw=2.8, ls="--", label="Simulation median")
    observed_mask = np.isfinite(observed_cases_series)
    ax.plot(dates[observed_mask], observed_cases_series[observed_mask], color="#34495e", lw=2.2,
            label="Reported cumulative confirmed cases")
    ax.set(title="Supplementary Figure S2: Cumulative cases across corrected simulations",
           ylabel="Cumulative cases", xlabel="Date (2026)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d5d8dc", ls="--", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.output_dir / "Supplementary_Figure_S2_pooled_review.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("S2 written.", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": "review-only regenerated S1/S2",
        "trajectories": args.trajectories, "horizon_days": HORIZON,
        "scenario": "no vaccination; base operations (30% detection, 30% tracing)",
        "initial_state": {"infectious": 15, "exposed": 15},
        "network_cache": str(args.network_cache), "network": "100,000 persons; household 5.2; community 30.0; variance 160.0",
        "rt": "one EpiNow2 posterior trajectory per LHS draw", "waiting_times": "Gamma; PSA means and shapes",
        "rt_source": args.rt_source,
        "allocator": "daily pooled onset cohort", "cpp_sha256": sha256(REPO / "scripts/production/ebola_stochastic_ring_cpp.cpp"),
        "wrapper_sha256": sha256(REPO / "scripts/production/ebola_stochastic_ring.py"),
        "fitted_parameters_sha256": sha256(fitted_path),
        "rt_posterior_sha256": sha256(posterior_path) if args.rt_source == "posterior" else None,
        "note": "S2 plots model onsets and reported confirmed cases together for calibration review; they are not identical surveillance quantities.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Review outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
