import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json
import os
import time
from paths import result_path

def seir_deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R, C = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dCdt = sigma * E
    return dSdt, dEdt, dIdt, dRdt, dCdt

def plot_publishable_calibration():
    # Set global font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    with open(result_path("fitted_parameters.json"), "r") as f:
        p = json.load(f)
        
    # Load incidence data
    df = pd.read_csv("BDBV2026-Data/data/insp_sitrep/processed/insp_sitrep__new_confirmed_cases__daily.csv")
    df['new_confirmed_cases'] = pd.to_numeric(df['new_confirmed_cases'], errors='coerce').fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    daily = df.groupby('date')['new_confirmed_cases'].sum().reset_index()
    daily['Day'] = (daily['date'] - daily['date'].min()).dt.days
    
    t_data = daily['Day'].values
    inc_data = daily['new_confirmed_cases'].values
    
    # 7-day rolling average for empirical
    rolling_inc = daily['new_confirmed_cases'].rolling(window=7, min_periods=1, center=True).mean().values
    
    # Parameters
    N_pop = p.get('N', 10000)
    E0 = p.get('E0', 20)
    I0 = p.get('I0', 10)
    gamma = p.get('gamma', 1.0 / p.get('infectious_period', 6.0))
    sigma = p.get('sigma', 1.0 / p.get('incubation_period', 8.5))
    
    # ODE Simulation
    beta = p['R0'] * gamma
    C0 = inc_data[0] if len(inc_data) > 0 else 1
    y0 = (N_pop - E0 - I0 - C0, E0, I0, 0, C0)
    
    t_smooth = np.linspace(0, max(t_data) + 10, 200)
    ret = odeint(seir_deriv, y0, t_smooth, args=(N_pop, beta, sigma, gamma))
    E_smooth = ret[:, 1]
    inc_rate_smooth = sigma * E_smooth
    
    # Rt Trajectory (from fitted_parameters.json or estimate_rt.py)
    rt_array = p.get('Rt_array', [p['R0']] * int(max(t_data)+10))
    t_rt = np.arange(len(rt_array))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Panel 1: Incidence
    ax1.bar(t_data, inc_data, color='#95A5A6', alpha=0.6, label="Reported Daily Incidence")
    ax1.plot(t_data, rolling_inc, color='#2C3E50', lw=2.5, label="7-Day Moving Average")
    ax1.plot(t_smooth, inc_rate_smooth, color='#E74C3C', lw=2.5, linestyle='--', label=f"SEIR Model Fit (Baseline R0 = {p['R0']:.2f})")
    
    ax1.set_ylabel("Daily Confirmed Cases", fontsize=12)
    ax1.set_title("A. Epidemiological Calibration", loc='left', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Rt
    ax2.plot(t_rt, rt_array, color='#2980B9', lw=2.5)
    ax2.axhline(1.0, color='#E74C3C', linestyle=':', lw=2)
    ax2.set_ylabel(r"Effective $R_t$", fontsize=12)
    ax2.set_xlabel("Days since outbreak detection", fontsize=12)
    ax2.set_title(r"B. Dynamic Reproduction Number ($R_t$)", loc='left', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(rt_array)*1.2)
    
    plt.tight_layout()
    
    timestamp = int(time.time())
    img_name = f"publishable_calibration_{timestamp}.png"
    img_path = os.path.join("figures", img_name)
    plt.savefig(img_path, dpi=300, facecolor='white')
    print(f"Saved High-Res Calibration to {img_path}")

if __name__ == "__main__":
    plot_publishable_calibration()
