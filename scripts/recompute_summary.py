import pandas as pd
import numpy as np

def summarize(arr):
    if len(arr) == 0:
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0}
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
    }

def main():
    print("Loading raw data...")
    raw = pd.read_csv("data_and_results/final_high_replicate_raw.csv")
    print(f"Loaded {len(raw)} rows.")
    
    summary_rows = []
    grouped = raw.groupby(["scenario", "level", "radius", "detection", "transmission_mode"])
    
    for name, group in grouped:
        cases = group["cases_percent"].values
        deaths = group["deaths_percent"].values
        vaccines = group["vaccines_percent"].values
        
        cases_summary = summarize(cases)
        deaths_summary = summarize(deaths)
        vaccine_summary = summarize(vaccines)
        
        first_row = group.iloc[0]
        
        # Add a check for "actual_vaccine_effect" to avoid KeyError
        ve = first_row.get("actual_vaccine_effect", 0.45)
        cfr = first_row.get("actual_vax_cfr", 0.454)
        
        summary_rows.append({
            "scenario": name[0],
            "level": name[1],
            "radius": name[2],
            "n": len(group),
            "population_size": first_row.get("population_size", 100000),
            "detection": name[3],
            "vaccine_effect": ve,
            "vax_cfr": cfr,
            "transmission_mode": name[4],
            "baseline_tau": first_row.get("baseline_tau", 0.25),
            "rt_max": first_row.get("rt_max", 6.06),
            "household_mean": first_row.get("household_mean", 5.0),
            "community_mean": first_row.get("community_mean", 5.0),
            "community_variance": first_row.get("community_variance", 25.0),
            "cases_percent_mean": cases_summary["mean"],
            "cases_percent_median": cases_summary["median"],
            "cases_percent_p25": cases_summary["p25"],
            "cases_percent_p75": cases_summary["p75"],
            "deaths_percent_mean": deaths_summary["mean"],
            "deaths_percent_median": deaths_summary["median"],
            "deaths_percent_p25": deaths_summary["p25"],
            "deaths_percent_p75": deaths_summary["p75"],
            "vaccines_percent_mean": vaccine_summary["mean"],
            "vaccines_percent_p25": vaccine_summary["p25"],
            "vaccines_percent_p75": vaccine_summary["p75"],
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("data_and_results/final_high_replicate_summary.csv", index=False)
    print("Done generating final_high_replicate_summary.csv")

if __name__ == "__main__":
    main()
