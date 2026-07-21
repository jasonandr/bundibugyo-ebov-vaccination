import csv

from paths import result_path


MODEL_PARAMETERS = [
    {
        "domain": "Network",
        "parameter": "Network topology",
        "main_value": "Two-layer household-community graph",
        "source_or_rationale": "Implemented in scripts/ebola_stochastic_ring.py",
        "sensitivity_or_notes": "Household/caregiving cliques plus overdispersed community contacts",
    },
    {
        "domain": "Network",
        "parameter": "Population size in main precomputed sweeps",
        "main_value": "2000 nodes",
        "source_or_rationale": "Computationally tractable Monte Carlo grid size",
        "sensitivity_or_notes": "Model can be run on larger networks; report as simulation scale, not health-zone population",
    },
    {
        "domain": "Natural history",
        "parameter": "Mean incubation period",
        "main_value": "8.5 days",
        "source_or_rationale": "BDBV/EVD natural-history assumption used in simulation scripts",
        "sensitivity_or_notes": "Vary in sensitivity analyses where computational budget allows",
    },
    {
        "domain": "Natural history",
        "parameter": "Mean infectious period",
        "main_value": "6.0 days",
        "source_or_rationale": "Simulation default and grid-generation scripts",
        "sensitivity_or_notes": "Transmission ceases after recovery or detection/isolation",
    },
    {
        "domain": "Transmission",
        "parameter": "Baseline transmission hazard",
        "main_value": "tau = 0.08 in primary grids",
        "source_or_rationale": "Scenario parameter used in precomputed grid scripts",
        "sensitivity_or_notes": "Empirical Rt arrays are generated separately for calibrated analyses",
    },
    {
        "domain": "Mortality",
        "parameter": "Baseline CFR",
        "main_value": "45.4% in model default and primary grids",
        "source_or_rationale": "Delay-adjusted estimate from latest public 2026 data",
        "sensitivity_or_notes": "Historical supplemental matrices use 25% and 51%",
    },
    {
        "domain": "Vaccination",
        "parameter": "Protection against disease",
        "main_value": "Varied across scenario grids",
        "source_or_rationale": "Unknown cross-protection against BDBV",
        "sensitivity_or_notes": "Main grids span low-to-high efficacy assumptions",
    },
    {
        "domain": "Vaccination",
        "parameter": "Therapeutic rescue",
        "main_value": "50% relative CFR reduction when fully active",
        "source_or_rationale": "Scenario assumption for post-exposure benefit",
        "sensitivity_or_notes": "Figure 6 and supplemental matrices vary this assumption",
    },
    {
        "domain": "Vaccination",
        "parameter": "Immune onset",
        "main_value": "10-day binary delay or sigmoidal onset",
        "source_or_rationale": "Implemented in simulate_ring_vaccination",
        "sensitivity_or_notes": "Fast, standard, and slow sigmoidal profiles evaluated",
    },
    {
        "domain": "Surveillance",
        "parameter": "Case detection/reporting probability",
        "main_value": "Varied across grids",
        "source_or_rationale": "Central operational bottleneck",
        "sensitivity_or_notes": "Figure 3 uses 40%, 60%, 80%; other grids sweep broader ranges",
    },
    {
        "domain": "Surveillance",
        "parameter": "Detection delay",
        "main_value": "4.0 days after symptom onset",
        "source_or_rationale": "Operational scenario assumption",
        "sensitivity_or_notes": "Delay-vs-uptake heatmaps evaluate this dimension",
    },
    {
        "domain": "Tracing",
        "parameter": "Radius 1 tracing delay",
        "main_value": "2.0 days after detection",
        "source_or_rationale": "Simulation default",
        "sensitivity_or_notes": "Queueing can extend realized vaccination time",
    },
    {
        "domain": "Tracing",
        "parameter": "Radius 2 added delay",
        "main_value": "2.0 additional days in default model",
        "source_or_rationale": "Operational penalty for contacts of contacts",
        "sensitivity_or_notes": "Figure 2B sweeps this parameter",
    },
    {
        "domain": "Tracing",
        "parameter": "Radius 2 uptake multiplier",
        "main_value": "0.75 by default",
        "source_or_rationale": "Reduced extended-ring coverage",
        "sensitivity_or_notes": "Figure 2B sweeps relative coverage",
    },
    {
        "domain": "Tracing",
        "parameter": "Maximum tracing bandwidth",
        "main_value": "100 or 1000 traces/day depending on scenario",
        "source_or_rationale": "Operational queue capacity",
        "sensitivity_or_notes": "Use scenario-specific value in figure captions/results",
    },
    {
        "domain": "Stockpile",
        "parameter": "Maximum vaccine courses",
        "main_value": "500, 1000, 3000 in Figure 3C",
        "source_or_rationale": "Resource-constrained stockpile scenarios",
        "sensitivity_or_notes": "Implemented mechanistically in event scheduler",
    },
    {
        "domain": "Uncertainty",
        "parameter": "Monte Carlo trials",
        "main_value": "10000 per scenario cell",
        "source_or_rationale": "Stochastic simulation uncertainty",
        "sensitivity_or_notes": "Reported intervals are Interquartile Range (IQR, 25th-75th percentiles)",
    },
]


def write_csv(rows):
    path = result_path("model_reporting_parameters.csv")
    fields = ["domain", "parameter", "main_value", "source_or_rationale", "sensitivity_or_notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(rows):
    path = result_path("model_reporting_parameters.md")
    with open(path, "w") as f:
        f.write("# Model Reporting Parameter Table\n\n")
        f.write("| Domain | Parameter | Main value | Source/rationale | Sensitivity/notes |\n")
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                "| {domain} | {parameter} | {main_value} | {source_or_rationale} | {sensitivity_or_notes} |\n".format(
                    **row
                )
            )
    return path


def main():
    csv_path = write_csv(MODEL_PARAMETERS)
    md_path = write_markdown(MODEL_PARAMETERS)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
