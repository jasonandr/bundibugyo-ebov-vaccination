import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gamma
from scipy.optimize import minimize
import os
import time

# Set nice aesthetic defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def estimate_rt_for_incidence(cases_inc_raw, rolling_window=7):
    cases_inc = pd.Series(cases_inc_raw).rolling(window=rolling_window, min_periods=1, center=False).mean().values
    
    mean_g = 15.3
    std_g = 9.3
    shape = (mean_g / std_g)**2
    scale = (std_g**2) / mean_g
    w = gamma.pdf(np.arange(1, len(cases_inc)+1), a=shape, scale=scale)
    w = w / np.sum(w)
    
    prior_mean = 1.5
    prior_sd = 1.0
    prior_shape = (prior_mean / prior_sd)**2
    prior_scale = (prior_sd**2) / prior_mean
    
    Rt_empirical = np.zeros_like(cases_inc)
    
    for t in range(len(cases_inc)):
        t_start = t
        def nll(rt_val):
            R = rt_val[0]
            if R <= 0: return 1e9
            log_prior = gamma.logpdf(R, a=prior_shape, scale=prior_scale)
            log_lik = 0.0
            for s in range(t_start, t + 1):
                Lambda_s = 0.0
                for tau in range(1, s + 1):
                    if tau <= len(w):
                        Lambda_s += cases_inc[s - tau] * w[tau - 1]
                mu_s = R * Lambda_s
                if mu_s <= 0:
                    if cases_inc[s] > 0: log_lik += -1e9
                    continue
                if cases_inc[s] == 0: log_lik += -mu_s
                else: log_lik += cases_inc[s] * np.log(mu_s) - mu_s
            return -(log_lik + log_prior)
        
        res = minimize(nll, x0=[prior_mean], bounds=[(0.01, 20.0)])
        Rt_empirical[t] = res.x[0]

    Rt_smooth = pd.Series(Rt_empirical).rolling(window=3, min_periods=1, center=True).mean().values
    return cases_inc, Rt_smooth

def generate_historical_plot(df_daily, cases_inc_raw, cases_inc_smooth, Rt_smooth, title, output_name, is_weekly=False):
    fig, ax3 = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    color_cases = '#2C3E50'
    color_rt = '#E74C3C'
    
    if is_weekly:
        df_weekly = df_daily.groupby(pd.Grouper(key='Date', freq='W')).agg({'Cases': 'sum'}).reset_index()
        ax3.bar(df_weekly['Date'], df_weekly['Cases'], color=color_cases, alpha=0.7, width=6.0, edgecolor='none', label='Weekly Confirmed Cases')
        ax3.set_ylabel("Weekly Confirmed Cases", fontsize=13, fontweight='bold', color=color_cases)
    else:
        ax3.bar(df_daily['Date'], cases_inc_raw, color=color_cases, alpha=0.7, width=1.0, edgecolor='none', label='Daily Confirmed Cases')
        ax3.set_ylabel("Daily Confirmed Cases", fontsize=13, fontweight='bold', color=color_cases)
    
    ax3.tick_params(axis='y', labelcolor=color_cases, labelsize=11)
    
    max_date = df_daily['Date'].max() + pd.Timedelta(days=2)
    ax3.set_xlim(pd.to_datetime(df_daily['Date'].min()) - pd.Timedelta(days=2), max_date)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    days_duration = (df_daily['Date'].max() - df_daily['Date'].min()).days
    interval = max(5, days_duration // 10)
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=11)
    
    ax3_rt = ax3.twinx()
    ax3_rt.plot(df_daily['Date'], Rt_smooth, color=color_rt, linewidth=3, label='Estimated $R_t$')
    ax3_rt.set_ylabel("Effective Reproduction Number ($R_t$)", fontsize=13, fontweight='bold', color=color_rt)
    ax3_rt.tick_params(axis='y', labelcolor=color_rt, labelsize=11)
    
    max_rt = np.nanmax(Rt_smooth)
    if np.isnan(max_rt) or max_rt <= 0: max_rt = 5.0
    ax3_rt.set_ylim(0, max_rt * 1.2)
    ax3_rt.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=2.0, alpha=0.8)
    
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.grid(axis='x', visible=False)
    ax3.spines['top'].set_visible(False)
    ax3_rt.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    img_path = os.path.join(out_dir, output_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {img_path}")
    return img_path

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    
    f2012 = os.path.join(root, "data_and_results", "BDBV_2012_Isiro_daily_cases.csv")
    df_2012 = pd.read_csv(f2012)
    df_2012['Date'] = pd.to_datetime(df_2012['Date'])
    df_2012 = df_2012.sort_values('Date')
    date_range_2012 = pd.date_range(start=df_2012['Date'].min(), end=df_2012['Date'].max(), freq='D')
    df_daily_2012 = pd.DataFrame({'Date': date_range_2012})
    df_daily_2012 = pd.merge(df_daily_2012, df_2012, on='Date', how='left').fillna(0)
    
    cases_inc_raw_2012 = df_daily_2012['Cases'].values
    cases_inc_smooth_2012, Rt_smooth_2012 = estimate_rt_for_incidence(cases_inc_raw_2012)
    
    timestamp = int(time.time())
    img2012 = f"figS2_cases_rt_2012_{timestamp}.png"
    generate_historical_plot(df_daily_2012, cases_inc_raw_2012, cases_inc_smooth_2012, Rt_smooth_2012, "", img2012, is_weekly=False)
    
    f2007 = os.path.join(root, "data_and_results", "BDBV_2007_Wamala_weekly_cases.csv")
    df_2007 = pd.read_csv(f2007)
    df_2007 = df_2007.sort_values('Epi_Week')
    
    start_date_2007 = pd.to_datetime('2007-07-09')
    
    daily_cases_2007 = []
    daily_dates_2007 = []
    
    current_date = start_date_2007
    for idx, row in df_2007.iterrows():
        weekly_cases = row['Cases']
        daily_val = weekly_cases / 7.0
        for d in range(7):
            daily_dates_2007.append(current_date + pd.Timedelta(days=d))
            daily_cases_2007.append(daily_val)
        current_date += pd.Timedelta(days=7)
        
    df_daily_2007 = pd.DataFrame({'Date': daily_dates_2007, 'Cases': daily_cases_2007})
    cases_inc_raw_2007 = df_daily_2007['Cases'].values
    cases_inc_smooth_2007, Rt_smooth_2007 = estimate_rt_for_incidence(cases_inc_raw_2007)
    
    img2007 = f"figS1_cases_rt_2007_{timestamp}.png"
    generate_historical_plot(df_daily_2007, cases_inc_raw_2007, cases_inc_smooth_2007, Rt_smooth_2007, "", img2007, is_weekly=True)
    
    out_json = os.path.join(root, "data_and_results", "historical_plot_files.json")
    with open(out_json, "w") as f:
        import json
        json.dump({"2007": img2007, "2012": img2012}, f)

if __name__ == "__main__":
    main()
