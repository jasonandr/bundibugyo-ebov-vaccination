# Model reproducibility ledger

**Status:** active reconstruction record  
**Last updated:** 2026-07-23 00:20 PDT (America/Los_Angeles)  
**Purpose:** single technical source of truth for the R1 analysis. This ledger distinguishes the manuscript-supported model specification from legacy or incompatible outputs. Update this file whenever a model input, scenario definition, run, table, or figure changes.

## Immediate decision

The primary analysis must be rerun using the updated EpiNow2 Rt curve before any numerical result is retained in the manuscript, Table 2, Figure 2, or supplementary figures. The first priority is the core strategy comparison that supports the main findings.

## Canonical production code and data locations

| Component | Canonical location | Status |
|---|---|---|
| C++ stochastic transmission engine | `scripts/production/ebola_stochastic_ring_cpp.cpp` | Use for all reruns |
| Python model wrapper and two-layer network generator | `scripts/production/ebola_stochastic_ring.py` | Use for all reruns |
| Reproduction-number estimation | `scripts/production/estimate_rt.py` | Use only if recalibration is required |
| Current outbreak data pipeline | `scripts/production/current_outbreak_data.py` | Source of notification data |
| Authoritative EpiNow2 output | `results/epinow_rt.csv` | **Use this run.** Median Rt starts at 2.635 on 14 May 2026 and corresponds to Figure 1/S1. |
| Calibrated median daily forcing | `data_and_results/fitted_parameters.json` → `Rt_array` | Exact 66-day median curve from the authoritative EpiNow2 output |
| `rt_posterior_samples.npy` | `data_and_results/rt_posterior_samples.npy` | **Do not use.** It predates the updated EpiNow2 run and begins at median Rt 1.647. |
| Figure renderer | `scripts/production/generate_all_main_figures.py` | Do not use for final tables/figures until it is refactored to read the new canonical result file |
| Updated Figure 2 PSA review output | `data_and_results/review_outputs/figure2_psa_updated_epinow_median_20260722/` | Current review candidate: updated EpiNow2 median Rt, 200 draws × 50 replicates |
| Prior Figure 2 PSA review output | `archive/2026-07-22_pre_github_cleanup/data_and_results_legacy/archive_debug_20260722/psa_and_calibration_checks/pooled_psa_figure2_200x50_20260722/` | **Invalidated:** used the obsolete posterior Rt array; retained only as an audit record |
| Reusable canonical network | `data_and_results/network_cache/production_network_20260722/network_000_seed_2026072501.npz` | Fixed 100,000-person 5.2/30.0/160.0 topology used for new review analyses |

## Locked model structure for the primary rerun

| Element | Specification | Evidence/status |
|---|---|---|
| Population | 100,000-person local transmission network | Manuscript and appendix |
| Network | Two-layer household/community network | Manuscript |
| Household mean degree | 5.2 | README and production configuration |
| Community mean degree | 30.0 | README and production configuration |
| Community-degree variance | 160.0 | README and production configuration |
| Legacy-named `baseline_tau` argument | Present in the wrapper for backwards compatibility; ignored by the active cohort allocator | Do not recalibrate or report it as an active transmission parameter |
| Time-varying transmission input | EpiNow2-derived daily `Rt_array` | `fitted_parameters.json` |
| Incubation period | mean 8.5 days | Manuscript Table 1 |
| Infectious period | mean 6.0 days | Manuscript Table 1 |
| Baseline CFR | 45.4% (delay-adjusted) | `fitted_parameters.json`; manuscript Table 1 |
| Vaccine effect | 45% base-case effect, applied to infection, post-exposure benefit, and mortality reduction | Manuscript and Figure 2 legend |
| Initial conditions | **15 infectious and 15 exposed** | Author-confirmed, locked specification for new production work |
| Horizon | 90 days for current production/PSA workflow | Must be fixed and documented in new runner |

## Operational scenario definitions

These are supported by the tracked manuscript and reviewer responses, not inferred from legacy output files.

| Operations | Index-case detection | Contact-tracing coverage | Use |
|---|---:|---:|---|
| Base operations | 30% | 30% | Base comparator |
| Enhanced operations | 70% | 80% | Strengthened case finding, contact investigation, and isolation |

Supporting record:

- Tracked manuscript: Base operations are explicitly defined as 30% detection and 30% tracing; Table 1 states 70% detection for enhanced operations and main vaccine comparisons.
- Reviewer responses: Enhanced operations are explicitly defined as 70% detection and 80% tracing.
- Supplementary appendix: enhanced operations increase detection and tracing before vaccination is added, allowing separation of surveillance/isolation effects from incremental vaccine effects.

## Primary scenario matrix to rerun

All scenarios below must be run on the same production code, network specification, calibrated parameter set, replicate design, horizon, and random-seed scheme. Each should retain raw replicate output and a scenario-level summary.

| ID | Strategy | Detection / tracing | Primary comparator |
|---|---|---|---|
| P0 | No vaccination, base operations | 30% / 30% | Reference |
| P1 | No vaccination, enhanced operations | 70% / 80% | P0 |
| P2 | Ring 1 vaccination, base operations | 30% / 30% | P0 |
| P3 | Ring 2 vaccination, base operations | 30% / 30% | P0 |
| P4 | Ring 1 vaccination, enhanced operations | 70% / 80% | P1 for incremental vaccine effect; P0 for total strategy effect |
| P5 | Ring 2 vaccination, enhanced operations | 70% / 80% | P1 for incremental vaccine effect; P0 for total strategy effect |
| P6 | Community vaccination, 20% coverage | 30% / 30% | P0 |
| P7 | Community vaccination, 40% coverage | 30% / 30% | P0 |
| P8 | Community vaccination, 60% coverage | 30% / 30% | P0 |
| P9 | Community vaccination, 80% coverage | 30% / 30% | P0 |

For the strategy-comparison figure, display total deaths averted versus P0. For the specific added-value-of-ring analysis, display incremental deaths averted versus P1. Do not mix these estimands in the same panel without explicit labels.

## Required outputs from the primary rerun

1. A raw, replicate-level file with scenario ID, seed, network seed/ID, calibrated-parameter version, all operational settings, cases, deaths, vaccine courses, and simulation horizon.
2. A scenario-level summary with expected cases, expected deaths, deaths averted, percent deaths averted, total doses, and doses per death averted.
3. A machine-readable manifest recording code hashes, exact input filenames, run date/time, number of replicates, and random-seed plan.
4. A regenerated Table 2, Figure 2, and Supplementary Figure S3 produced solely from the new scenario-level summary.
5. Verification that the no-vaccination base scenario is on the intended manuscript outcome scale before running the complete scenario matrix.
6. Regenerated Figure 3 trajectories and Figure 4 contour grids from the same production model, network, calibrated parameters, and comparator definitions.

## Outputs that must not be used for the current primary results

| Artifact | Reason |
|---|---|
| Archived `final_high_replicate_raw_*.csv` and `final_high_replicate_summary_*.csv` | Generated by `archive/2026-07-22_pre_github_cleanup/scripts_legacy/archive_exploratory/run_final_high_replicate_estimates.py`, which imports the legacy model wrapper and explicitly uses community mean 5.0 and variance 25.0. Incompatible with the current 30.0/160.0 production network. |
| `data_and_results/psa_raw_simulations.csv` and `psa_summary_results.csv` | Archived exploratory PSA output on an outcome scale inconsistent with the tracked manuscript Table 2. |
| `data_and_results/table_1_outbreak_outcomes.csv` | Older result table with a 1,900-death baseline; incompatible with the current manuscript scale. |
| Earlier uncorrected dose-efficiency review figure (`dose_efficiency_psa_updated_epinow_median_20260722`) | Ring 1 was compared with a Radius 2 no-vaccination comparator; superseded by the radius-matched PSA below. |

## Current manuscript/documentary record

| Document | Relevant content |
|---|---|
| `R1/Manuscript_R1_2026_07_19_track.docx` | Defines base operations (30%/30%), comparator logic, 45% vaccine effect, network size, and primary outcomes. |
| `R1/Lancet_ID_Reviewer_Responses.docx` | Defines enhanced operations (70% detection/80% tracing) and the rationale for those targets. |
| `R1/Supplementary_Appendix_R1_2026_07_19_track.docx` | Defines tracing, Ring 1/Ring 2 mechanics, community rollout, and comparator logic. |

## Change log

| Date/time (PT) | Update | Consequence |
|---|---|---|
| 2026-07-22 09:46 | Identified that archived PSA and final-high-replicate outputs use incompatible model workflows and/or network parameterisations. | Do not use them for current manuscript results. |
| 2026-07-22 09:46 | Documentary audit established base operations = 30% detection / 30% tracing; enhanced operations = 70% detection / 80% tracing. | Lock these values in the canonical runner. |
| 2026-07-22 09:46 | Production-engine diagnostic confirmed that initial conditions must be scaled from the calibrated population; it was a diagnostic only, not a final estimate. | Full primary rerun required with documented scenario definitions and replicate plan. |
| 2026-07-22 09:50 | Confirmed that Figures 3 and 4 are derived from archival scripts and outputs. SCG is available for a production-only rerun. | Rerun Figure 3 trajectories and all Figure 4 grids; do not retain existing values without revalidation. |
| 2026-07-22 10:07 | Submitted isolated SCG P0 production validation (job 52201391); it failed at compilation because SCG's default GCC 4.8.5 lacks C++17 support. | No simulations ran and no outputs were created. Batch environment updated to load GCC 13.3.0 before resubmission. |
| 2026-07-22 10:12 | SCG P0 validation job 52202150 compiled but stopped before simulation because the Anaconda Python environment exposes an older dynamic C++ runtime than GCC 13.3.0 requires. | No outputs were created. The batch build now statically links the C++ runtime to eliminate this environment-specific ABI conflict before one further validation submission. |
| 2026-07-22 10:36 | Cancelled SCG validation job 52202869 at the author's direction. It used the production C++ engine but regenerated a NetworkX graph for each replicate, rather than the intended paired shared-network workflow. | Do not use its partial/no output. Replace with a runner that fixes each network across its matched strategy simulations and records network and simulation seeds separately. |
| 2026-07-22 10:36 | Inspected the completed prior SCG 100-array run. Its runner explicitly specifies community mean 5.0 and variance 25.0 and regenerates a graph inside each simulation call. | These completed SCG outputs remain legacy artifacts and cannot be reused for the 30.0/160.0 production manuscript results. The intended historical shared-network runner still needs to be located or reconstructed from an authoritative design record. |
| 2026-07-22 10:45 | Added `scripts/production/run_paired_network_primary.py` and passed a local smoke test (one 100,000-person 5.2/30.0/160.0 network; one matched replicate across P0–P9). | The runner fixes the network within each network ID, uses the same simulation seed for each matched strategy comparison, retains both seed types, writes a manifest, and refuses to overwrite prior output. A one-network, 100-replicate production validation has started locally; it is not yet a manuscript result. |
| 2026-07-22 10:47 | Completed the one-network, 100-replicate paired validation (1,000 rows across P0–P9). The base-operations mean was 1,369 deaths (median 1,318), whereas the current Table 2 record gives 333 deaths (95% UI 280–385). | Do not scale to 100 networks or update any manuscript result. The discrepancy requires reconciliation of the cohort seeding, operational-call settings, and result-generating workflow before a full production rerun. |
| 2026-07-22 10:55 | Verified the active production C++ branch. With `engine="cpp"`, the wrapper selects the cohort branch: `Rt_array` is used directly and the legacy-named `baseline_tau` argument is inactive. | Do not recalibrate tau. The present engine assigns a binomial number of infections with expected value approximately `Rt(t)` for each newly infectious person, then samples susceptible neighbours; it does not aggregate all infectious people into an exact daily `I(t) × Rt(t)` cohort allocation. |
| 2026-07-22 10:55 | Audited archived Figure 3/4 direct-C++ generators. They use the 30.0/160.0 topology and direct `Rt_array`, but specify 50 infectious plus 50 exposed seeds and pass a vaccine-acceptability sentinel that bypasses the provided tracing-coverage array. | Existing Figure 3/4 outputs cannot establish the current operational scenarios and must be regenerated from the paired, wrapper-mediated production runner after the primary-call specification is locked. |
| 2026-07-22 11:02 | Confirmed the intended pooled daily cohort allocator is represented in `modify_cpp2.py`, but this change was never applied to any current/committed C++ engine. The live engine instead allocates expected `Rt(t)` infections separately for each newly infectious individual. | The one-network paired validation is invalid for primary inference and must not be scaled or used. Port and test the pooled allocator in a new production source before rerunning. |
| 2026-07-22 11:02 | Author confirmed the intended initial state is 15 infectious and 15 exposed individuals. | Replace the erroneous scaled 7/8 seed calculation in the new paired runner only after the pooled production engine is implemented and validated. |
| 2026-07-22 11:06 | Ported the daily onset-cohort pooled allocator into `scripts/production/ebola_stochastic_ring_cpp.cpp`; the paired runner now uses 15 infectious plus 15 exposed seeds. A small time-series test verified pooled daily allocation, and a one-network, 100-replicate primary run completed. | The corrected P0 median was 1,231 deaths (95% empirical interval 598–1,744), still inconsistent with current Table 2 (333; 280–385). Hold the full 100-network run pending reconciliation of the Table 2 result-generating workflow or remaining scenario-call differences. |
| 2026-07-22 11:15 | Completed a corrected Figure 2 PSA pilot: 20 Latin-hypercube draws × 25 matched replicates × 8 strategies (4,000 calls), with posterior `Rt` trajectory draws, gamma waiting times, 15/15 seeds, a fixed 30/160 network, and active tracing coverage. | Pilot outputs are isolated in `data_and_results/pooled_psa_figure2_pilot_20260722/`. They are for parameter/specification review only, not manuscript replacement values. A first summary incorrectly compared enhanced-plus-ring to enhanced operations; this was corrected from the unchanged raw output, with the incremental-ring estimand retained as a separate row. |
| 2026-07-22 11:22 | Identified a definite archived PSA ring-operations bug. `run_lhs_psa_pipeline.py` passes `vaccine_acceptability=-1` to the direct C++ call; in that state the C++ engine bypasses the supplied tracing-coverage array and instead uses `uptake=0.8` for Radius 1 and `0.8 × 0.75=0.6` for Radius 2. | Archived PSA ring estimates labelled as 30% base tracing or sampled enhanced tracing did not implement those tracing assumptions. This plausibly inflated historical ring impact. The exact contribution cannot be separated from the simultaneous correction to pooled daily onset-cohort allocation. |
| 2026-07-22 11:22 | Latest authoritative production code for new PSA work is `scripts/production/ebola_stochastic_ring_cpp.cpp` (daily onset-cohort pooled allocator), `scripts/production/ebola_stochastic_ring.py` (wrapper), and `scripts/production/run_pooled_psa_figure2_pilot.py` (Figure 2 PSA workflow). | The compiled local extension is a build artifact only. Archived code and output files must not be used to generate replacement manuscript results. |
| 2026-07-22 11:19 | Validated the parallel PSA runner with two LHS draws on two workers, then launched the corrected full Figure 2 PSA locally: 200 LHS draws × 50 matched stochastic replicates × 8 strategies on eight workers (10,000 simulations per strategy). | Results are being written only to `data_and_results/pooled_psa_figure2_200x50_20260722/`; do not use them in manuscript files until completion and review. |
| 2026-07-22 11:28 | Completed the corrected full Figure 2 PSA: 80,000 raw calls, 200 parameter-draw expected values per strategy, and a Figure-2-ready summary. Community-vaccination expected mortality reductions were 46.0%, 70.5%, 84.2%, and 91.4% at 20%, 40%, 60%, and 80% coverage; these closely reproduce the historical community pattern. | Corrected ring estimates are materially smaller: 7.6% for Ring 2 under base operations and 3.1% incremental benefit beyond enhanced operations. Treat these as review results pending scientific adjudication; do not replace manuscript values automatically. |
| 2026-07-22 11:35 | Created the immutable canonical network cache `data_and_results/network_cache/production_network_20260722/network_000_seed_2026072501.npz` (3,531,826 directed adjacency entries). | It can be reused without regenerating the network for PSA diagnostics and figure reruns. The completed Figure 2 PSA used the same deterministic seed before this cache was persisted. |
| 2026-07-22 11:42 | Regenerated review-only Supplementary Figures S1 and S2 from 100 corrected pooled-cohort trajectories. Arrays and figures are in `data_and_results/pooled_supplementary_calibration_100_20260722/`. | S1 tracks the realised model `Rt` against posterior EpiNow2 inputs; S2 tracks cumulative simulated onset cases against reported cumulative confirmed cases. Neither replaces a submission-ready figure pending scientific review. |
| 2026-07-22 12:05 | Author identified the authoritative updated EpiNow2 calibration displayed in Figure 1/S1. Inspection confirmed `results/epinow_rt.csv` and `fitted_parameters.json:Rt_array` start at Rt=2.635, whereas `rt_posterior_samples.npy` starts at median Rt=1.647 and predates the update. | Invalidated the 11:28 Figure 2 PSA and 11:42 S1/S2 review outputs. New runs use the updated 66-day EpiNow2 median forcing. Matching posterior draws must be exported from the updated EpiNow2 fit before posterior-Rt uncertainty is added back to the PSA. |
| 2026-07-22 12:10 | Generated a 100-trajectory updated-Rt calibration check in `data_and_results/review_outputs/supplementary_calibration_updated_epinow_median_20260722/`. | With the updated median forcing, median realised Rt declines from 2.46 at day 0 to 1.43 at day 30 and 1.26 at day 65; the simulated median cumulative onset count at day 90 is 2,257 (95% range 1,144–3,995). Review S1/S2 before restarting the full scenario matrix. |
| 2026-07-22 12:15 | Updated S2 calibration review figure to omit DRC-wide reported cases, because those notifications describe the whole country while the model network contains 100,000 people. | Current S2 review file: `figures/review_updated_epinow_20260722/Supplementary_Figure_S2_pooled_review.png`. |
| 2026-07-22 12:18 | Completed the updated Figure 2 PSA rerun using the verified 66-day EpiNow2 median forcing: 200 LHS draws × 50 matched stochastic replicates × 8 strategies (80,000 calls). | Outputs are in `data_and_results/review_outputs/figure2_psa_updated_epinow_median_20260722/`. Median expected base-case deaths were 1,136 (95% range across draws 718–1,933). This is the current Figure 2 review candidate, not yet a manuscript replacement. |
| 2026-07-22 12:22–12:30 | Completed the matched 10-strategy dose-efficiency PSA: Ring 1 and Ring 2 vaccination under both base and enhanced operations, plus 20%, 40%, 60%, and 80% community vaccination. | Output directory is `data_and_results/review_outputs/dose_efficiency_psa_updated_epinow_median_20260722/`. The design is 200 LHS draws × 50 stochastic replicates per strategy on the cached 100,000-person 5.2/30.0/160.0 network, with the updated EpiNow2 median Rt forcing, 15 infectious and 15 exposed seeds, and gamma-distributed waiting times. |
| 2026-07-22 12:30 | Rendered a review-only three-panel dose-efficiency figure from the expanded PSA. | Ring 1 under base operations had a median **increase** of 108 deaths versus base operations (IQR 60 to 166 more deaths), so its doses-per-death-averted value is not estimable. This is retained in the figure rather than converted to a misleading ratio. Review artifact: `data_and_results/review_outputs/dose_efficiency_psa_updated_epinow_median_20260722/Dose_Efficiency_Review.png`. |
| 2026-07-22 12:36 | Diagnosed the apparent harmful Ring 1 result as a comparator-definition bug. The engine uses `ring_radius` for contact tracing/monitoring as well as vaccine reach. The Ring 1 vaccination scenario was incorrectly compared with a no-vaccination Radius 2 tracing scenario. | The Radius 2 comparator monitors an additional contact layer and therefore has stronger non-vaccine control. A matched 100-replicate fixed-parameter check with Radius 1 in both arms produced 49 mean deaths averted (median paired reduction 39.5) by Ring 1 vaccination. Ring 1 PSA results and the review dose-efficiency figure are invalid pending a targeted rerun with Radius 1 no-vaccination comparators. Ring 2 comparisons remain radius-matched. |
| 2026-07-22 12:45 | Completed the corrected radius-matched dose-efficiency PSA: 200 draws × 50 matched replicates × 12 raw scenarios (120,000 calls). | New source output: `data_and_results/review_outputs/dose_efficiency_psa_radius_matched_20260722/`. Ring 1 vaccination is now compared with Radius 1 no-vaccination base operations; Ring 2 and community strategy definitions are unchanged. The radius-matched review figure supersedes the prior dose-efficiency review artifact. |
| 2026-07-22 13:05 | Added and launched `scripts/production/run_current_figure_batches.py` for the remaining main-text and appendix simulation matrices. | The batch produces raw replicate files for Figure 3 operational, community-VE, and risk-compensation grids; Figure 4 rollout-delay and immune-onset analyses; Figure 5 rollout grid; Supplementary S3 delivery timing; and Supplementary S4 independent infection/mortality VE grids. All use the current pooled allocator, cached 5.2/30.0/160.0 network, updated EpiNow2 median Rt, 15/15 initial state, gamma waiting times, and explicit tracing coverage. Outputs are isolated pending completion and rendering. |
| 2026-07-22 13:12 | Completed the 31,500-call current-engine batch and rendered review figures. | Figures 3A/B, Figure 4 rollout delay, Figure 5, and Supplementary S3/S4 have raw current-engine outputs. Do **not** submit the current Figure 3C risk-compensation panel: the pooled cohort allocator does not currently apply `risk_compensation_multiplier`. Also rerun the immune-onset panel with matched seeds before use, because independent seeds by midpoint produce avoidable Monte Carlo noise. |
| 2026-07-22 13:25 | Audited the first Figure 3 rerender after restoring risk compensation. | The first implementation incorrectly multiplied exposure for every vaccinated susceptible contact. The prior cohort-model definition multiplies onward transmission from vaccinated breakthrough infectious people; production code now implements that source-based definition. The Figure 3 grids also require a paired-comparator rerun: the initial batch used independent seeds by grid cell, yielding impossible variation in no-vaccination rows (eg, coverage=0 across nominal VE values). Do not use the current Figure 3 review image. |
| 2026-07-22 14:34 | Implemented the author-specified cohort-weighted risk-compensation rule in `scripts/production/ebola_stochastic_ring_cpp.cpp`: vaccinated infectious sources receive the multiplier; vaccinated eligible contacts receive the multiplier in both the cohort expected-infection calculation and the weighted target allocation. | This supersedes the prior source-only and target-only risk implementations. Figure 3C must use results generated by this compiled production engine. |
| 2026-07-22 14:52 | Added `scripts/production/run_paired_figure3c_figure4c.py` and completed a 4,900-row same-seed paired Figure 3C run. | The base-operations no-vaccination comparator uses the same cached network and random seed for every cell. The multiplier 1.0 / VE 0% cell is exactly 0% mortality reduction, eliminating the prior independent-comparator artifact. |
| 2026-07-22 14:55 | Re-ran Figure 4C with common random seeds across onset values and 1,000 same-seed paired replicates per point. | The updated medians are 8.6%, 8.6%, 8.1%, and 6.7% mortality reduction for immune-onset midpoints of 5, 7, 10, and 14 days, respectively. Use these values rather than the earlier 100-replicate independent-seed panel. |
| 2026-07-22 14:55 | Consolidated current review renders; moved superseded, smoke-test, and exploratory Figure 5 artifacts to the dated archive. | Figure 5 is not an active manuscript figure and is omitted from the current review directory. No Dropbox manuscript figure was overwritten. |
| 2026-07-22 15:15 | Revised the risk-compensation sensitivity after audit of the source-and-contact weighted implementation. The current display uses a contact-only multiplier: vaccinated eligible contacts are weighted in the pooled cohort infection target and in destination sampling; vaccinated infectious sources are not multiplied. | The full paired raw results remain retained for audit. The review display is intentionally limited to a plausible, interpretable range (risk multiplier 1.0–2.0; vaccine effectiveness 15–90%) and is a sensitivity analysis, not a primary effectiveness estimate. |
| 2026-07-22 15:25 | Replaced the obsolete three-panel immune-onset/timing Figure 4 concept with a paired two-panel implementation analysis. Panel A varies community-vaccination coverage; Panel B varies campaign initiation at declaration, +14 days, and +28 days. | This removes the uninformative immune-onset panels. Raw paired output: `data_and_results/review_outputs/CURRENT_REVIEW_20260722/raw/community_coverage_delay_0_14_28_paired_20260722.csv`. |
| 2026-07-22 15:58 | Created the consolidated review-only package, now located at `figures/current_review/manuscript_review_figures_20260722/`. | It contains candidate PNG renders for Figures 1–4 and the updated dose-efficiency figure. The Dropbox submission directory was not modified. |
| 2026-07-22 16:00 | Re-rendered the radius-matched dose-efficiency figure without interval bars, placed all numeric labels immediately beyond their bar ends, and removed the parameter-draw footer. | Current source and package copy: `data_and_results/review_outputs/dose_efficiency_psa_radius_matched_20260722/Dose_Efficiency_Radius_Matched_Review.png` and `figures/current_review/manuscript_review_figures_20260722/Supplementary_Figure_S3_Dose_Efficiency_review.png`. |
| 2026-07-22 16:20 | Performed a GitHub-oriented repository cleanup. Current code, inputs, review outputs, figures, and manuscript documents are retained in their dedicated top-level folders; legacy simulations, manuscript-editing scripts, obsolete figures, scratch files, and unused external repositories were moved intact to `archive/2026-07-22_pre_github_cleanup/`. | The active workflow is now limited to `scripts/production/`, `data_and_results/`, `figures/current_review/`, `manuscript/current/`, and the required `BDBV2026-Data/` submodule. No active result file was deleted. |
| 2026-07-23 00:20 | Reconciled the Table 2 outcome-scale discrepancy and updated all three R1 documents. The tracked manuscript's 333-death base case was traced to the obsolete 5/25-network, old-Rt, per-individual-allocator workflow; the author adopted the corrected 30/160-network, updated-EpiNow2, pooled-allocator values (base-operations median 1,136 expected deaths [95% range 718–1,933 across 200 draws]). | Manuscript, supplement, and reviewer responses were edited as tracked changes on working copies in `manuscript/working_2026_07_23/` (edit scripts in `scripts/manuscript_tools/`, values in `numbers_concordance.md`). Table 2, Summary, Methods (PSA design 200×50, 15/15 seeds, 90-day horizon, Rt hold-constant), Results, Discussion, Table 1, figure captions, supplement model description, new Tables S2/S3, and all stale response quotes were updated. Outstanding items are listed in `manuscript/working_2026_07_23/HANDOFF_NOTES.md`. Dropbox files were not modified. |
| 2026-07-23 | Author-directed follow-up work: (1) Main-text Table 2 moved to Supplementary Table S4; dose-efficiency figure promoted to main-text Figure 5 (whiskers render, `render_dose_efficiency_psa_main.py`). (2) Extinction analysis: 4 strategies × 500 paired reps (`run_extinction_paired.py`, outputs in `review_outputs/extinction_paired_20260723/`): early extinction before day 90 occurred in 0% of base-operations and community-40%/60% simulations and 9.2% (95% CI 7.0–12.1) of enhanced-operations simulations (median day 84). (3) Immune-onset regenerated at 1,000 paired reps (`review_outputs/immune_onset_paired_20260723/`): 8.56/8.58/8.08/6.72% at midpoints 5/7/10/14 — matches the ledger's lost 14:55 values. (4) Historical robustness rerun with production engine (`run_historical_robustness.py`, 2 settings × 4 strategies × 625 reps, `review_outputs/historical_robustness_production_20260723/`): 2007 — enhanced 93.2%, ring incremental 43.8%, community-40% 97.1%; 2012 — 75.7%, 10.2%, 81.5%; same ordering as archived Table S1. (5) Reference overhaul: removed Althaus/WHO-ERT/Brewer-duplicate, added Lhomme NEJM 2026, Abbott EpiNow2, Kuppalli Lancet ID 2026; previously uncited refs (Keeling, Zomahoun, Mooring, Wells-conflict, Mayhew) now cited; full renumber with citation remap verified 1–30. (6) Submission figure folder refreshed (Figures 2–4 corrected renders, new Figure 5, S2–S5 assembled). | Documents updated as tracked changes; responses now quote the new metrics. The compiled extension `scripts/production/ebola_stochastic_ring_cpp.cpython-313-darwin.so` was rebuilt from current production source for these runs (agents 1–3). |
| 2026-07-23 | Figure refinement round (author requests): (a) Independent-VE grid rerun finer (`run_s4_independent_ve.py`, 13×13 cells × 100 paired reps + base arm = 17,000 sims, `review_outputs/s4_independent_ve_fine_20260723/`); verified the ve_m→vax_CFR mapping empirically (ve_i=0, ve_m=0 cell matches the no-vax base arm exactly per paired replicate; median deaths monotone in both VE components, zero violations) — the mortality-VE mechanism works correctly. (b) S4 re-rendered (`render_s4_independent_ve.py`): sequential 0–100 YlGnBu (no negative range), cubic interpolation + Gaussian smoothing, no on-figure title. (c) S2/S3/S5 re-rendered without on-figure titles (`render_supplementary_s2_s3_s5.py`); S5 is now two panels (A: sigmoid protection profiles with step comparator; B: 1,000-rep midpoint bars without value labels). (d) Supplement S5 caption updated; all renders copied to the Dropbox `Figures_Submission_Ready/` folder. | Supplementary figures S2–S5 are current-engine, title-free, and consistent with the locked model specification. Grid max median mortality reduction 78.4% at VE 90/90 (no 80% contour exists). |

## Current consolidated review figures

The following files are the single review package for visual inspection as of 22 July 2026. They are not automatically submission-ready and must be promoted to Dropbox only after scientific and manuscript review.

| Package file | Source | Definition/status |
|---|---|---|
| `Figure_1_review.png` | Current Dropbox submission-ready Figure 1, copied without modification | Displays the updated EpiNow2-derived outbreak input. |
| `Figure_2_review.png` | `review_outputs/figure2_psa_updated_epinow_median_20260722/Figure_2_updated_epinow_review_ordered.png` | 200 LHS parameter draws × 50 matched stochastic replicates per strategy; updated 66-day EpiNow2 median Rt forcing. Review candidate only. |
| `Figure_3_review.png` | `review_outputs/CURRENT_REVIEW_20260722/Figure_3_weighted_risk_review.png` | Operational and VE grids use the current production model. Panel C is the paired contact-only risk-compensation sensitivity described above. |
| `Figure_4_review.png` | `review_outputs/CURRENT_REVIEW_20260722/Figure_4_review.png` | Community-vaccination implementation timing and coverage, based on paired simulations with starts at declaration, +14, and +28 days. |
| `Supplementary_Figure_S3_Dose_Efficiency_review.png` | `review_outputs/dose_efficiency_psa_radius_matched_20260722/Dose_Efficiency_Radius_Matched_Review.png` | 200 LHS draws × 50 matched replicates; Ring 1 comparisons use Radius 1 no-vaccination comparators, and Ring 2/community comparisons use Radius 2/base comparators as appropriate. No error bars are displayed. |

The package README records the same file-level provenance: `figures/current_review/manuscript_review_figures_20260722/README.md`.
