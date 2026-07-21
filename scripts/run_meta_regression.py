import pandas as pd
import numpy as np
import statsmodels.api as sm
from paths import result_path

df = pd.read_csv(result_path("targeted_psa_results.csv"))

# Calculate averted cases (Base - Strategy)
df["ring_averted"] = df["base_cases_abs"] - df["ring_cases_abs"]
df["comm_averted"] = df["base_cases_abs"] - df["comm_cases_abs"]

# Aggregate over replicates to get the mean per parameter set
df_agg = df.groupby("set_id").mean().reset_index()

params = [
    "incubation_period", "infectious_period", 
    "baseline_tau", "reporting_rate", "vaccine_effect",
    "incubation_shape", "infectious_shape"
]

# Standardize features for standardized beta coefficients
X = df_agg[params]
X_std = (X - X.mean()) / X.std()
X_std = sm.add_constant(X_std)

y_ring = df_agg["ring_averted"]
y_ring_std = (y_ring - y_ring.mean()) / y_ring.std()

model_ring = sm.OLS(y_ring_std, X_std).fit()
print("=== Meta-Regression for Ring Vaccination Averted Cases ===")
print(model_ring.summary())

