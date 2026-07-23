# Bundibugyo ebolavirus vaccination modelling

Simulation code, data, and outputs for *Ring and community vaccination for
Bundibugyo ebolavirus outbreak response: a stochastic network modelling study*.

The package contains a stochastic network transmission model of Bundibugyo
ebolavirus (BDBV) and the code required to reproduce the study's simulation
experiments and figures: case-detection and contact-tracing scenarios,
reactive ring vaccination, community vaccination, and probabilistic
sensitivity analyses.

## Model overview

- Individual-based SEIR model on a two-layer contact network (100,000
  persons): fully connected household/caregiving clusters (DRC DHS
  household-size distribution, mean ≈5.1) and a negative-binomial community
  layer (mean degree 30.0, variance 160.0).
- Time-varying transmission follows a daily effective reproduction number
  Rt(t) estimated from outbreak notifications with a Bayesian renewal model
  (Gamma generation interval, mean 15.3 days, SD 9.3 days), allocated through
  a pooled daily onset-cohort process. Rt is held constant at its final
  estimated value beyond the 66-day estimation window. The simulation horizon
  is 90 days, initialized with 15 infectious and 15 exposed individuals.
- Gamma-distributed incubation (mean 8.5 days) and infectious (mean 6.0 days)
  periods; baseline case fatality risk 45.4%.
- Operational scenarios: base operations (30% case detection, 30% contact
  tracing) and enhanced operations (70% detection, 80% tracing); reactive
  ring vaccination (Ring 1 and Ring 2); community vaccination at 20–80%
  coverage.
- Principal analyses use 200 Latin-hypercube parameter draws with 50 matched
  stochastic replicates per strategy (10,000 simulations per strategy).

## Requirements

Python 3.13, the packages in `requirements.txt`, a C++17 compiler, and
pybind11. Public outbreak notification data are provided via the
`BDBV2026-Data` submodule.

## Building

```bash
git submodule update --init BDBV2026-Data
cd scripts/production
c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
  ebola_stochastic_ring_cpp.cpp $(python3-config --embed --ldflags) \
  -o "ebola_stochastic_ring_cpp$(python3-config --extension-suffix)"
```

## Reproducing analyses

Simulation runners in `scripts/production/` write raw replicate-level output,
summaries, and a manifest (parameters, random seeds, code hashes, run date)
to `data_and_results/outputs/`. Runners take an explicit output directory
and refuse to overwrite existing outputs. Renderers in the same directory
regenerate the figures in `figures/final/`.

| Runner | Analysis |
|---|---|
| `run_pooled_psa_figure2_pilot.py` | Figure 2 probabilistic sensitivity analysis (200 draws × 50 replicates) |
| `run_figure3_paired_grids.py` | Figure 3 operational, vaccine-effect, and risk-compensation grids |
| `run_figure4_community_timing_paired.py` | Figure 4 coverage and initiation-timing analyses |
| `run_extinction_paired.py` | Early-extinction analysis (4 strategies × 500 paired replicates) |
| `run_paired_figure3c_figure4c.py` | Risk-compensation and immune-onset paired analyses |
| `run_historical_robustness.py` | Historical robustness (2007 and 2012 outbreaks) |
| `run_s4_independent_ve.py` | Independent infection/mortality vaccine-efficacy grid |
| `build_production_network_cache.py` | Build the cached 100,000-person network |
| `generate_pooled_supplementary_calibration.py` | Calibration trajectories (Supplementary Figure S2) |
| `estimate_rt.py` | Reproduction-number estimation from notification data |

## Repository layout

```text
scripts/production/    Model engine, network generator, runners, renderers
data_and_results/      Model inputs and simulation outputs
  fitted_parameters.json  Calibrated parameters (Rt trajectory, CFR)
  network_cache/          Fixed 100,000-person network and its manifest
  outputs/                Simulation outputs with run manifests
figures/final/         Figures 1–5 and Supplementary Figures S1–S5
BDBV2026-Data/         Public outbreak notification data (submodule)
```

## Data availability

Public outbreak notification data are included via the `BDBV2026-Data`
submodule. All simulation outputs needed to reproduce the figures are
included in `data_and_results/outputs/`.
