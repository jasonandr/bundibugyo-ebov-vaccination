import pandas as pd
import glob
import os

def merge_csvs():
    raw_files = [f"data_and_results/final_high_replicate_raw_{i}.csv" for i in range(1, 21)]
    missing = [f for f in raw_files if not os.path.exists(f)]
    if missing:
        print(f"Missing files: {missing}")
        return

    print(f"Found all 20 raw files. Merging...")
    df_raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    df_raw.to_csv("data_and_results/final_high_replicate_raw.csv", index=False)
    print(f"Wrote data_and_results/final_high_replicate_raw.csv with {len(df_raw)} rows.")
    
if __name__ == "__main__":
    merge_csvs()
