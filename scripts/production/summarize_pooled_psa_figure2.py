"""Create Figure-2 expected-value summaries from a pooled PSA raw file."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    raw_path = args.output_dir / "raw_replicates.csv"
    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["draw_id"]), row["scenario"]), []).append(row)
    draw_values = []
    for draw_id in sorted({key[0] for key in grouped}):
        base_deaths = float(np.mean([float(r["deaths"]) for r in grouped[(draw_id, "no_vax_base_ops")]]))
        ring1_base_deaths = (float(np.mean([float(r["deaths"]) for r in grouped[(draw_id, "no_vax_ring1_base_ops")]]))
                             if (draw_id, "no_vax_ring1_base_ops") in grouped else base_deaths)
        enhanced_deaths = float(np.mean([float(r["deaths"]) for r in grouped[(draw_id, "no_vax_enh_ops")]]))
        for scenario in sorted(s for d, s in grouped if d == draw_id):
            group = grouped[(draw_id, scenario)]
            comparator_deaths = ring1_base_deaths if (scenario.startswith("ring1_") or scenario == "no_vax_ring1_enh_ops") else base_deaths
            mean_deaths = float(np.mean([float(r["deaths"]) for r in group]))
            row = {"draw_id": draw_id, "scenario": scenario,
                   "mean_cases": float(np.mean([float(r["cases"]) for r in group])),
                   "mean_deaths": mean_deaths,
                   "mean_doses": float(np.mean([float(r["doses"]) for r in group])),
                   "comparator_mean_deaths": comparator_deaths,
                   "deaths_averted_pct": 100 * (comparator_deaths - mean_deaths) / comparator_deaths}
            draw_values.append(row)
            if scenario == "vax_enh_ops":
                draw_values.append({**row, "scenario": "incremental_ring_vax",
                                    "comparator_mean_deaths": enhanced_deaths,
                                    "deaths_averted_pct": 100 * (enhanced_deaths - mean_deaths) / enhanced_deaths})
    with (args.output_dir / "draw_expected_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(draw_values[0])); writer.writeheader(); writer.writerows(draw_values)
    summary = []
    for scenario in sorted({r["scenario"] for r in draw_values}):
        values = np.array([r["deaths_averted_pct"] for r in draw_values if r["scenario"] == scenario])
        summary.append({"scenario": scenario, "n_draws": len(values),
                        "median_deaths_averted_pct": float(np.median(values)),
                        "ui_low_95": float(np.quantile(values, 0.025)),
                        "ui_high_95": float(np.quantile(values, 0.975)),
                        "mean_deaths_averted_pct": float(np.mean(values))})
    with (args.output_dir / "figure2_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)


if __name__ == "__main__":
    main()
