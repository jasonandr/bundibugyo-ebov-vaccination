import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def plot_bdbv_history():
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Check if files exist
    f2007 = "data_and_results/BDBV_2007_Wamala_weekly_cases.csv"
    f2012 = "data_and_results/BDBV_2012_Isiro_daily_cases.csv"
    
    timestamp = int(time.time())
    out_dir = "figures/"
    
    if os.path.exists(f2007):
        df_2007 = pd.read_csv(f2007)
        if len(df_2007) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            # assuming columns like Date or Week, and Cases
            cols = df_2007.columns
            x_col = cols[0]
            y_col = cols[1]
            ax.bar(df_2007[x_col], df_2007[y_col], color='maroon')
            ax.set_title("2007 Bundibugyo Outbreak (Uganda)", fontsize=14)
            ax.set_ylabel("Cases")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"BDBV_2007_outbreak_{timestamp}.png"))
            plt.close()
            
    if os.path.exists(f2012):
        df_2012 = pd.read_csv(f2012)
        if len(df_2012) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            cols = df_2012.columns
            x_col = cols[0]
            y_col = cols[1]
            ax.bar(df_2012[x_col], df_2012[y_col], color='navy')
            ax.set_title("2012 Isiro Outbreak (DRC)", fontsize=14)
            ax.set_ylabel("Cases")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"BDBV_2012_outbreak_{timestamp}.png"))
            plt.close()

if __name__ == "__main__":
    plot_bdbv_history()
