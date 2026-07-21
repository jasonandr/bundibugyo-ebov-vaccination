import pandas as pd
import sys
import os

main_df = pd.read_csv("data_and_results/final_high_replicate_raw.csv")

missing_dfs = []
for i in range(25):
    f = f"data_and_results/final_high_replicate_raw_scg_dilution_{i}.csv"
    if os.path.exists(f):
        missing_dfs.append(pd.read_csv(f))
    else:
        print(f"Warning: {f} not found!")

if len(missing_dfs) == 25:
    merged_missing = pd.concat(missing_dfs, ignore_index=True)
    
    # We shouldn't duplicate rows if they already exist, but for simplicity we can just concat.
    # To be safe, let's drop any old rows for these 3 specific scenarios and then append the new ones.
    scenarios_to_replace = ["stockpile_cap_radius1", "stockpile_cap_radius2_protected", "stockpile_cap_radius2_competing"]
    
    main_df = main_df[~main_df["scenario"].isin(scenarios_to_replace)].copy()
    main_df = pd.concat([main_df, merged_missing], ignore_index=True)
    
    main_df.to_csv("data_and_results/final_high_replicate_raw.csv", index=False)
    print("Merged 25 dilution chunks successfully!")
else:
    print(f"Only found {len(missing_dfs)} chunks. Not merging yet.")
    sys.exit(1)
