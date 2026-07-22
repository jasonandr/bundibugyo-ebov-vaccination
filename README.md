# Bundibugyo ebolavirus vaccination modelling

Reproducibility package for the revised manuscript, *Ring and community
vaccination for Bundibugyo ebolavirus outbreak response: a stochastic network
modelling study*.

## Active repository layout

```text
scripts/production/                 Production model, runners, and renderers
data_and_results/                   Locked inputs, network cache, and current outputs
figures/current_review/             Consolidated Figures 1–4 and dose-efficiency render
manuscript/current/                 Current tracked manuscript, appendix, and responses
BDBV2026-Data/                      Public notification-data submodule
MODEL_REPRODUCIBILITY_LEDGER.md     Dated technical decisions, checks, and provenance
PSA_PARAMETER_CONCORDANCE_AUDIT_2026-07-22.md
                                   Locked PSA specification and audit trail
archive/2026-07-22_pre_github_cleanup/
                                   Superseded code, outputs, draft documents, and scratch work
```

The `archive/` directory is intentionally excluded from Git. It is retained
locally for audit and recovery, but must not be used to regenerate manuscript
results.

## Locked current model specification

- 100,000-person two-layer network: household mean 5.2; community mean 30.0;
  community-degree variance 160.0.
- Daily onset-cohort pooled allocation driven by the 66-day updated EpiNow2
  median `Rt_array` in `data_and_results/fitted_parameters.json`.
- Initial state: 15 infectious and 15 exposed persons; 90-day horizon.
- Gamma-distributed incubation and infectious waiting times.
- Base operations: 30% detection and 30% contact tracing. Enhanced operations:
  70% detection and 80% contact tracing, with the documented ramp.
- Figure 2 and dose-efficiency PSA: 200 Latin-hypercube parameter draws × 50
  matched stochastic replicates per strategy.

The authoritative detailed record is
[`MODEL_REPRODUCIBILITY_LEDGER.md`](MODEL_REPRODUCIBILITY_LEDGER.md).

## Reproduce the current review outputs

Install the Python dependencies in `requirements.txt`, initialise the public
data submodule, and compile the pybind extension from the production source.

```bash
git submodule update --init BDBV2026-Data
cd scripts/production
c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
  ebola_stochastic_ring_cpp.cpp $(python3-config --embed --ldflags) \
  -o "ebola_stochastic_ring_cpp$(python3-config --extension-suffix)"
```

The active runners accept explicit output paths and refuse to overwrite prior
artifacts. Current inputs and result manifests are retained under
`data_and_results/review_outputs/`. The consolidated visual package is in
`figures/current_review/manuscript_review_figures_20260722/`.

## Current candidate figures

- Figure 1: updated EpiNow2 outbreak input and model schematic.
- Figure 2: updated-EpiNow2 PSA forest plot.
- Figure 3: paired operational and community-effect grids; Panel C is the
  contact-only risk-compensation sensitivity.
- Figure 4: paired community-vaccination coverage and initiation timing.
- Supplementary Figure S3: radius-matched dose-efficiency comparison.

All figures in this package are review candidates, not automatically promoted
to the submission directory.
