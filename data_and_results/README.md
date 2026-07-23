# Data and results

- `fitted_parameters.json` — calibrated model parameters, including the
  66-day daily effective reproduction number (Rt) trajectory and the
  delay-adjusted baseline case fatality risk (45.4%).
- `sitrep_2026_07_05_override.csv` — notification-data update used by the
  outbreak-data loader.
- `network_cache/production_network_20260722/` — fixed 100,000-person
  two-layer network used for all principal analyses, with its manifest.
- `outputs/` — simulation outputs. Each directory corresponds to one
  analysis and contains raw replicate-level results, summaries, and a
  manifest (parameters, random seeds, code hashes, run date).
