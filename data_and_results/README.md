# Data and results directory

This directory is deliberately organised around active reproducibility, not chronology.

## Use now

- `fitted_parameters.json` — the locked 66-day updated EpiNow2 median Rt forcing.
- `sitrep_2026_07_05_override.csv` — reviewed notification-data update used by the production loader.
- `network_cache/production_network_20260722/` — reusable fixed 100,000-person 5.2/30/160 network cache.
- `review_outputs/figure2_psa_updated_epinow_median_20260722/` — current Figure 2 PSA review candidate (80,000 calls, manifest included).
- `review_outputs/CURRENT_REVIEW_20260722/` — current Figure 3–4 raw inputs and renders.
- `review_outputs/dose_efficiency_psa_radius_matched_20260722/` — current radius-matched dose-efficiency PSA.
- `review_outputs/supplementary_calibration_updated_epinow_median_20260722/` — updated-Rt calibration trajectories and review figures.

## Preserved but not current

- All historical, debug, smoke-test, and superseded outputs are retained in
  `../archive/2026-07-22_pre_github_cleanup/data_and_results_legacy/`.

The pre-22 July legacy high-replicate 5/25-network output set and obsolete PSA/sweep outputs were permanently deleted at the author's request. Do not use any archive output to update the manuscript without a new documented review.
