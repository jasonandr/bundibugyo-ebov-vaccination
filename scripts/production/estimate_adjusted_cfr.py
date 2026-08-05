import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os
import time
import json

from paths import result_path
from current_outbreak_data import cumulative_confirmed_cases, cumulative_confirmed_deaths

def estimate_delay_adjusted_cfr():
    # 1. Load 2026 Data
    cases = cumulative_confirmed_cases()
    deaths = cumulative_confirmed_deaths()
    
    df = pd.merge(cases, deaths, on=['Country', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Enforce monotonicity
    for col in ['Cases', 'Deaths']:
        arr = df[col].values
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                arr[i] = arr[i-1]
        df[col] = arr
        
    df['Incident_Cases'] = df['Cases'].diff().fillna(df['Cases'].iloc[0])
    
    # 2. Get Delay Distribution from 2012 Linelist
    ll = pd.read_csv('BDBV2026-Data/data/bdbv-linelist-analysis/data/linelist.csv')
    ll['Date_of_onset_symp'] = pd.to_datetime(ll['Date_of_onset_symp'])
    ll['Date_of_Death'] = pd.to_datetime(ll['Date_of_Death'])
    dead = ll[ll['Outcome'] == 'Dead'].copy()
    dead['delay'] = (dead['Date_of_Death'] - dead['Date_of_onset_symp']).dt.days
    delays = dead['delay'].dropna().values
    
    # Fit gamma to delays
    # mean = 11.4, median = 10.0 -> shape ~ k, scale ~ theta
    # using scipy.stats
    fit_alpha, fit_loc, fit_beta = gamma.fit(delays, floc=0)
    
    # 3. Calculate Adjusted CFR (Nishiura method)
    # Expected cases with known outcomes by day T = sum( incidence(t) * CDF(T-t) )
    t_days = (df['Date'] - df['Date'].min()).dt.days.values
    
    adjusted_cfr = []
    naive_cfr = []
    expected_known_outcomes = []
    
    for i, row in df.iterrows():
        # Naive
        if row['Cases'] > 0:
            naive_cfr.append(row['Deaths'] / row['Cases'])
        else:
            naive_cfr.append(np.nan)
            
        # Adjusted
        # For each past day j (0 to i), how many cases have had time to die?
        # That is incidence[j] * P(delay <= i - j)
        expected_resolved = 0
        for j in range(i + 1):
            days_passed = i - j
            p_resolved = gamma.cdf(days_passed, fit_alpha, scale=fit_beta)
            expected_resolved += df['Incident_Cases'].iloc[j] * p_resolved
            
        expected_known_outcomes.append(expected_resolved)
        
        if expected_resolved > 0:
            adjusted_cfr.append(row['Deaths'] / expected_resolved)
        else:
            adjusted_cfr.append(np.nan)
            
    df['Naive_CFR'] = naive_cfr
    df['Adjusted_CFR'] = adjusted_cfr
    
    # 4. Plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Naive_CFR'] * 100, 'b--', linewidth=2, label='Naïve CFR (Deaths / Cumulative Cases)')
    plt.plot(df['Date'], df['Adjusted_CFR'] * 100, 'r-', linewidth=3, label='Delay-Adjusted CFR (Nishiura Method)')
    
    # 2012 baseline
    plt.axhline(53.8, color='k', linestyle=':', linewidth=2, label='2012 Isiro Outbreak Final CFR (53.8%)')
    
    plt.title('Bundibugyo 2026: Naïve vs. Delay-Adjusted Case Fatality Rate', fontsize=14, fontweight='bold')
    plt.ylabel('Case Fatality Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    out_dir = "figures"
    timestamp = int(time.time())
    img_name = f"adjusted_cfr_{timestamp}.png"
    img_path = os.path.join(out_dir, img_name)
    plt.savefig(img_path, dpi=300)
    print(f"Latest Naive CFR: {df['Naive_CFR'].iloc[-1]:.1%}")
    print(f"Latest Adjusted CFR: {df['Adjusted_CFR'].iloc[-1]:.1%}")
    print(f"Saved to {img_path}")
    print(f"FILENAME:{img_name}")

    try:
        with open(result_path("fitted_parameters.json"), "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        params = {}
    latest = df.iloc[-1]
    latest_adjusted = float(latest["Adjusted_CFR"])
    params.update({
        "latest_data_date": str(latest["Date"].date()),
        "latest_confirmed_cases": int(latest["Cases"]),
        "latest_confirmed_deaths": int(latest["Deaths"]),
        "latest_naive_cfr": float(latest["Naive_CFR"]),
        "latest_adjusted_cfr": latest_adjusted,
        "base_CFR": latest_adjusted,
        "vax_CFR": latest_adjusted * 0.5,
    })
    with open(result_path("fitted_parameters.json"), "w") as f:
        json.dump(params, f, indent=4)

if __name__ == "__main__":
    estimate_delay_adjusted_cfr()
