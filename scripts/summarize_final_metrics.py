import pandas as pd
import numpy as np

raw = pd.read_csv("data_and_results/final_high_replicate_raw.csv")
baseline_df = raw[raw["scenario"] == "analysis_1_reactive_ring"].copy()

baseline_cases = baseline_df[baseline_df["level"] == "no_vax_base_ops"].set_index(['seed'])['cases_percent'].to_dict()
baseline_deaths = baseline_df[baseline_df["level"] == "no_vax_base_ops"].set_index(['seed'])['deaths_percent'].to_dict()

def cases_averted(row):
    base = baseline_cases.get(row['seed'], 8.0)
    if base == 0: return 0.0
    return max(0.0, (base - row['cases_percent']) / base * 100.0)

def deaths_averted(row):
    base = baseline_deaths.get(row['seed'], 4.0)
    if base == 0: return 0.0
    return max(0.0, (base - row['deaths_percent']) / base * 100.0)

raw["Cases_Averted"] = raw.apply(cases_averted, axis=1)
raw["Deaths_Averted"] = raw.apply(deaths_averted, axis=1)

grouped = raw.groupby(["scenario", "level"])
with open("data_and_results/final_abstract_metrics.txt", "w") as f:
    for name, group in grouped:
        med_deaths = group["Deaths_Averted"].median()
        p25_deaths = group["Deaths_Averted"].quantile(0.25)
        p75_deaths = group["Deaths_Averted"].quantile(0.75)
        f.write(f"{name[0]} - {name[1]}: {med_deaths:.1f}% ({p25_deaths:.1f}-{p75_deaths:.1f}%)\n")
