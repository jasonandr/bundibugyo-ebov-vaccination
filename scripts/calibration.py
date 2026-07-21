import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import json
from paths import result_path

def seir_deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R, C = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dCdt = sigma * E
    return dSdt, dEdt, dIdt, dRdt, dCdt

def run_calibration():
    # Load data
    df = pd.read_csv("BDBV2026-Data/build/long/insp_sitrep__national_cumulative_confirmed_cases.csv", header=None, names=['Country', 'Date', 'Cases'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days
    
    t_data = df['Day'].values
    cases_cum = df['Cases'].values.copy()
    
    # Enforce monotonicity: if cases drop due to data cleaning/reclassification,
    # back-propagate the correction so cumulative cases never decrease.
    for i in range(len(cases_cum)-2, -1, -1):
        if cases_cum[i] > cases_cum[i+1]:
            cases_cum[i] = cases_cum[i+1]
            
    # Calculate interval incidence 
    cases_inc = np.diff(cases_cum)
    
    N = 1000000 
    sigma = 1.0 / 8.5
    gamma = 1.0 / 6.0
    
    C0 = cases_cum[0]
    
    def objective(params):
        R0, E0, I0 = params
        if R0 < 1.0 or R0 > 5.0 or E0 < 0 or I0 < 0:
            return 1e9
        
        beta = R0 * gamma
        y0 = (N - E0 - I0 - C0, E0, I0, 0, C0)
        
        # We need the model predictions at the exact days we have data
        ret = odeint(seir_deriv, y0, t_data, args=(N, beta, sigma, gamma))
        C_pred = ret[:, 4]
        
        C_pred_inc = np.diff(C_pred)
        
        # Fit to incidence (daily/interval cases) instead of cumulative
        sse = np.sum((C_pred_inc - cases_inc)**2)
        return sse

    # Initial guess
    initial_guess = [1.5, 10.0, 5.0]
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    
    best_R0, best_E0, best_I0 = result.x
    print(f"Calibration to Incidence successful!")
    print(f"Fitted R0: {best_R0:.2f}")
    print(f"Fitted Initial Exposed (E0): {best_E0:.2f}")
    print(f"Fitted Initial Infected (I0): {best_I0:.2f}")
    
    with open(result_path("fitted_parameters.json"), "w") as f:
        json.dump({"R0": best_R0, "E0": best_E0, "I0": best_I0, "sigma": sigma, "gamma": gamma, "N": N}, f)

if __name__ == "__main__":
    run_calibration()
