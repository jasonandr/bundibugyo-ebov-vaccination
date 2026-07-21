import pandas as pd
import glob
import os

def merge_csvs():
    raw_files = glob.glob("data_and_results/final_high_replicate_raw_*.csv")
    if not raw_files:
        print("No raw files found.")
        return

    print(f"Found {len(raw_files)} raw files.")
    df_raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    df_raw.to_csv("data_and_results/final_high_replicate_raw.csv", index=False)
    print("Wrote data_and_results/final_high_replicate_raw.csv")
    
    # We don't actually need to merge summary files since we can just recalculate them by running
    # run_final_high_replicate_estimates.py locally again (which reads the merged raw file).
    # This is much safer to ensure summaries are 100% correct across chunks.

if __name__ == "__main__":
    merge_csvs()
