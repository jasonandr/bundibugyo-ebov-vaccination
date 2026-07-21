import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

from paths import figure_path, result_path

def run_estimation():
    # Load data
    df_susp = pd.read_csv('BDBV2026-Data/data/insp_sitrep/processed/insp_sitrep__new_suspected_cases__daily.csv')
    df_conf = pd.read_csv('BDBV2026-Data/data/insp_sitrep/processed/insp_sitrep__new_confirmed_cases__daily.csv')
    
    # Replace 'ND' with NaN and drop them for aggregation
    df_susp['new_suspected_cases'] = pd.to_numeric(df_susp['new_suspected_cases'], errors='coerce')
    df_conf['new_confirmed_cases'] = pd.to_numeric(df_conf['new_confirmed_cases'], errors='coerce')
    
    # Ensure dates are datetime
    df_susp['date'] = pd.to_datetime(df_susp['date'])
    df_conf['date'] = pd.to_datetime(df_conf['date'])
    
    # Aggregate nationally by date
    daily_susp = df_susp.groupby('date')['new_suspected_cases'].sum().reset_index()
    daily_conf = df_conf.groupby('date')['new_confirmed_cases'].sum().reset_index()
    
    # Merge
    merged = pd.merge(daily_susp, daily_conf, on='date', how='outer').fillna(0).sort_values('date')
    
    # Set index to fill missing dates if any
    merged.set_index('date', inplace=True)
    merged = merged.asfreq('D', fill_value=0)
    
    # 7-day rolling sums to calculate Test Positivity smoothly
    merged['rolling_susp'] = merged['new_suspected_cases'].rolling(window=7, min_periods=1).sum()
    merged['rolling_conf'] = merged['new_confirmed_cases'].rolling(window=7, min_periods=1).sum()
    
    # Calculate TP
    # If rolling_susp is 0, TP = 0 or we forward fill
    merged['test_positivity'] = np.where(
        merged['rolling_susp'] > 0,
        merged['rolling_conf'] / merged['rolling_susp'],
        np.nan
    )
    merged['test_positivity'] = merged['test_positivity'].ffill().fillna(0.5) # Assume bad detection before testing scales up
    
    # Map TP to Detection Rate
    # Anchor 1: TP >= 50% -> Detection = 10%
    # Anchor 2: TP <= 5% -> Detection = 80%
    # Linear interpolation in between
    tp_high = 0.50
    dr_low = 0.10
    tp_low = 0.05
    dr_high = 0.80
    
    slope = (dr_high - dr_low) / (tp_low - tp_high)
    
    def map_tp_to_dr(tp):
        if tp >= tp_high: return dr_low
        if tp <= tp_low: return dr_high
        return dr_low + slope * (tp - tp_high)
        
    merged['detection_rate'] = merged['test_positivity'].apply(map_tp_to_dr)
    
    # Smooth detection rate (14 day rolling to avoid jagged behavior in simulation)
    merged['smoothed_detection'] = merged['detection_rate'].rolling(window=14, min_periods=1, center=True).mean().bfill()
    
    # Create the array for the simulator
    detection_array = merged['smoothed_detection'].tolist()
    
    # Ensure it reaches the length of the simulation (150+ days). If empirical data is shorter, we plateau.
    while len(detection_array) < 300:
        detection_array.append(detection_array[-1])
        
    # Save to JSON
    with open(result_path('detection_array.json'), 'w') as f:
        json.dump({'detection_array': detection_array}, f)
        
    print(f"Generated detection array of length {len(detection_array)}. Min: {min(detection_array):.3f}, Max: {max(detection_array):.3f}")
    
    # Plot to verify
    plt.figure(figsize=(10, 6))
    plt.plot(merged.index, merged['test_positivity'], label='7-Day Rolling Test Positivity', color='red', alpha=0.6)
    plt.plot(merged.index, merged['smoothed_detection'], label='Inferred Detection Rate', color='blue', linewidth=2.5)
    plt.title("Empirical Scale-up of Surveillance Ascertainment")
    plt.ylabel("Fraction")
    plt.xlabel("Date")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(figure_path("empirical_detection_curve.png"), dpi=150)
    
if __name__ == "__main__":
    run_estimation()
