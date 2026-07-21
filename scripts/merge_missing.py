import pandas as pd
import sys
import os

main_df = pd.read_csv("data_and_results/final_high_replicate_raw.csv")

missing_dfs = []
for i in range(5):
    f = f"data_and_results/final_high_replicate_raw_scg_missing_{i}.csv"
    if os.path.exists(f):
        missing_dfs.append(pd.read_csv(f))
    else:
        print(f"Warning: {f} not found!")

if missing_dfs:
    merged_missing = pd.concat(missing_dfs, ignore_index=True)
    main_df = pd.concat([main_df, merged_missing], ignore_index=True)
    main_df.to_csv("data_and_results/final_high_replicate_raw.csv", index=False)
    print("Merged 5 missing VE scenarios successfully!")
else:
    print("No missing files found to merge.")
