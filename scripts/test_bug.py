import numpy as np
import json
import ebola_stochastic_ring_cpp as sim
from ebola_stochastic_ring import generate_network

G = generate_network(100000)

with open("data_and_results/fitted_parameters.json") as f:
    RT_ARRAY = json.load(f)["Rt_array"]

deaths = []
for i in range(50):
    res = sim.simulate_mechanism_cpp(
        N=100000,
        adj=G,
        rt_array=RT_ARRAY,
        baseline_tau=0.25,
        incubation_period=8.5,
        infectious_period=6.0,
        ring_radius=0,
        efficacy=0.0,
        immune_delay=10.0,
        uptake=0.0,
        reporting_rate=0.7,
        reporting_rate_scalar=1.0,
        detection_delay=4.0,
        tracing_delay=2.0,
        max_cases=0,
        max_daily_traces=100,
        max_vaccines=0,
        base_CFR=0.454,
        vax_CFR=0.454,
        initial_infected=5,
        initial_exposed=0,
        max_sim_time=300,
        vax_start_time=0.0,
        return_time_series=False,
        risk_compensation_multiplier=1.0,
        trust_uptake_dependency=False,
        tracing_coverage=[],
        vaccine_acceptability=-1.0,
        sigmoidal_k=0.5,
        sigmoidal_d0=10.0,
        uptake_r2_drop=0.75,
        tracing_delay_r2_add=2.0,
        compete_queue=False,
        allow_pep=True,
        community_vax_coverage=0.0,
        community_vax_trigger=0,
        community_vax_delay=0.0,
        seed=i,
        infection_efficacy_multiplier=1.0,
        immediate_mortality_protection=False
    )
    deaths.append(res["total_deaths"])

print("Median deaths:", np.median(deaths))
