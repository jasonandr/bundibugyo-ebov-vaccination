# PSA parameter concordance audit

**Timestamp:** 2026-07-22 12:05 PDT  
**Purpose:** Lock the intended PSA inputs before regenerating Figure 2 values. This is an audit record only; it does not amend any manuscript or submitted figure.

## Documentary specification

| Domain | Intended specification supported by manuscript/reviewer response |
|---|---|
| Transmission | Daily `Rt(t)` from the updated EpiNow2 output in `results/epinow_rt.csv` / `fitted_parameters.json:Rt_array`; Gamma generation interval, mean 15.3 days and SD 9.3 days. The curve starts at Rt=2.635 on 14 May 2026. |
| Network | 100,000 people; household mean 5.2; community mean 30.0; community-degree variance 160.0. |
| Initial state | 15 infectious and 15 exposed. |
| Natural history | Gamma waiting times. Baseline means: incubation 8.5 days; infectious period 6.0 days. |
| Baseline operations | 30% detection; 30% tracing; detection delay 4 days; Radius-1 tracing delay 2 days. |
| Enhanced operations | Detection target 70%; tracing target 80%; 15-day ramp from baseline targets; detection delay 2 days. |
| Vaccine | Base efficacy 45%, applied jointly to infection protection, post-exposure benefit, and mortality reduction; sigmoid midpoint 10 days; gamma timing distributions. |
| Community vaccination in Table 2/Figure 2 PSA | 20%, 40%, 60%, and 80% coverage under **base operations**, immediate start, 14-day rollout; fixed doses of 20,000, 40,000, 60,000, and 80,000. |
| Estimand | For each parameter draw, average matched stochastic replicates within each strategy, then calculate deaths averted relative to the stated comparator. The Figure 2 interval is the 2.5th–97.5th percentile across parameter-draw expected values—not a raw single-outbreak prediction interval. |

## Archived PSA implementation (`run_lhs_psa_pipeline.py`)

| Parameter | Archived sampler/code | Concordance status |
|---|---|---|
| Vaccine efficacy | Uniform 0.30–0.60 | Consistent with a PSA around 45%; confirm final range before submission. |
| Rt | Nominally samples 1,000 posterior trajectories; silently falls back to the median `Rt_array` if posterior draws are absent. | Must make posterior-input availability a hard requirement, not a silent fallback. |
| Incubation mean | Normal(8.5, 1.0), sampled between its 5th and 95th percentiles. | Consistent with baseline mean; gamma waiting-time shape also varied. |
| Infectious mean | Normal(6.0, 0.8), sampled between its 5th and 95th percentiles. | Consistent with baseline mean; gamma waiting-time shape also varied. |
| Detection delays | Base Uniform(3,5); enhanced Uniform(1.5,3.5). | Compatible with the documented 4-day and 2-day baselines. |
| Enhanced tracing | Uniform(0.60,0.90). | Intended range is reasonable, but the archived direct C++ call passes `vaccine_acceptability=-1`, which bypasses the supplied tracing array. **Bug: not applied.** |
| Community variance | Uniform(80,240). | **Bug: not applied.** The archived pipeline generates one fixed 30/160 network before sampling. |
| Gamma shapes | Uniform(1,3) for incubation and infectious periods. | Applied by the C++ gamma distributions. |
| Initial state | 15 infectious + 15 exposed. | Correct. |
| Horizon | 90 days. | Correct. |
| Population/network | 100,000; 5.2/30/160. | Correct for the fixed network actually generated. |
| Stochastic replicates | 50 per LHS draw. | Produces expected-value PSA summaries, but differs from the stated 10,000 total simulations per scenario unless draw and replicate counts are reconciled. |
| C++ allocator | The archived compiled engine predates the daily onset-cohort pooled allocator. | **Not acceptable for regenerated PSA.** |

## Required corrected pilot

1. Use the new production pooled allocator and gamma waiting times.
2. Use 15/15 seeds, 90 days, 5.2/30/160 network, and max 100 daily traces.
3. Use wrapper-mediated calls so tracing coverage is active; set vaccine acceptability separately to 1.0.
4. Sample the archived PSA distributions except community variance, which cannot vary without regenerating a network for each draw. In the initial Figure 2 pilot, hold variance at the manuscript value of 160.0 and label this limitation.
5. Run the Figure 2/Table 2 scenario set with matched simulation seeds within parameter draw.
6. Write draw-level expected values and a Figure-2-ready summary CSV. Do not overwrite `psa_summary_results.csv` or any submission figure.

## Confirmed archived ring-operations bug (2026-07-22 11:22 PDT)

The archived PSA passes `vaccine_acceptability=-1` in its direct C++ call. In the engine, the supplied `tracing_coverage` array is used only when `vaccine_acceptability >= 0`; otherwise, tracing defaults to the general uptake parameter. Consequently, the archived PSA operated reactive rings at 80% Radius-1 reach and 60% Radius-2 reach, even when rows were labelled as 30% base tracing or sampled enhanced tracing. The corrected workflow sets `vaccine_acceptability=1.0` and passes tracing coverage explicitly, separating contact reach from vaccine acceptance.

## Latest production paths

- Pooled transmission engine: `scripts/production/ebola_stochastic_ring_cpp.cpp`
- Python wrapper: `scripts/production/ebola_stochastic_ring.py`
- Figure 2 PSA runner: `scripts/production/run_pooled_psa_figure2_pilot.py`
- Figure 2 PSA summarizer: `scripts/production/summarize_pooled_psa_figure2.py`
- Run ledger: `MODEL_REPRODUCIBILITY_LEDGER.md`

## Superseded review outputs (2026-07-22 12:05 PDT)

The prior Figure 2 PSA review output is self-contained in
`data_and_results/pooled_psa_figure2_200x50_20260722/`:

- `raw_replicates.csv`: 80,000 corrected simulation calls (200 LHS draws × 50 matched replicates × 8 strategies).
- `draw_expected_values.csv`: the draw-level expected outcomes used to form uncertainty intervals.
- `figure2_values.csv`: the Figure 2-ready deaths-averted summary (2.5th–97.5th percentiles over draw-level expected values).
- `manifest.json`: code/input hashes, network seed, initial state, and run design.

The reusable fixed topology for subsequent review runs is
`data_and_results/network_cache/production_network_20260722/network_000_seed_2026072501.npz`, with its companion manifest. These outputs are **superseded**: they used `rt_posterior_samples.npy`, whose median initial Rt is 1.647 and which predates the updated EpiNow2 calibration.

## Locked Rt rule for the rerun

Use the 66-day median daily trajectory in `data_and_results/fitted_parameters.json:Rt_array`, which exactly matches the updated EpiNow2 median in `results/epinow_rt.csv`. Do not use `data_and_results/rt_posterior_samples.npy`. The updated EpiNow2 raw posterior draws are not saved in the repository, so the immediate rerun treats the updated median forcing as fixed; this preserves the correct calibration without fabricating an unverified posterior sample.

## Updated EpiNow2-median PSA output (2026-07-22 12:18 PDT)

`data_and_results/review_outputs/figure2_psa_updated_epinow_median_20260722/` is the current Figure 2 review candidate. It contains 80,000 raw simulation calls, draw-level expected outcomes, a Figure-2-ready summary, and a manifest. It uses 200 LHS draws × 50 matched replicates per strategy, the 15/15 initial state, the pooled onset-cohort allocator, the fixed 5.2/30/160 network, gamma waiting times, and the updated EpiNow2 median Rt forcing. It does not incorporate Rt posterior uncertainty because the verified updated posterior draws are not available locally.
