import pandas as pd
from paths import result_path

print("Loading main high replicate raw data...")
main_df = pd.read_csv(result_path("final_high_replicate_raw.csv"))

# Drop the old bad stockpile cap scenarios
print("Dropping old 500, 1000, 3000 stockpile cap rows...")
main_df = main_df[~((main_df["scenario"] == "stockpile_cap") & (main_df["level"].astype(str).isin(["500", "1000", "3000"])))]

print("Loading new stockpile caps from SCG array...")
stockpile_dfs = []
for i in range(6):
    try:
        df = pd.read_csv(result_path(f"final_high_replicate_raw_scg_array_stockpile_new_{i}.csv"))
        stockpile_dfs.append(df)
    except FileNotFoundError:
        print(f"Warning: Missing part {i}")
stockpile_df = pd.concat(stockpile_dfs, ignore_index=True)
stockpile_df["level"] = stockpile_df["level"].astype(str)

print("Concatenating and saving...")
final_df = pd.concat([main_df, stockpile_df], ignore_index=True)
final_df.to_csv(result_path("final_high_replicate_raw.csv"), index=False)
print("Saved final_high_replicate_raw.csv with corrected stockpile caps!")
