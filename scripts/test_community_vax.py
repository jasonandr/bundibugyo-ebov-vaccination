import networkx as nx
from ebola_stochastic_ring import simulate_ring_vaccination, generate_network

print("Testing pure Python engine...")
G = generate_network(N=1000)

# 1. Test Day 0 Vaccination (Trigger 1) with 50% coverage
res_cases, res_deaths, res_vax = simulate_ring_vaccination(
    G, 
    ring_radius=0, # Turn off ring
    community_vax_coverage=0.5,
    community_vax_trigger=1, # Day 0
    community_vax_delay=0.0,
    engine='python'
)
print(f"Python Day 0 Vax | Vaccines delivered: {res_vax} (Expected ~500)")
assert 480 <= res_vax <= 520, "Python Day 0 vaccination count is off"


# 2. Test First Detection Vaccination (Trigger 2) with 75% coverage
res_cases, res_deaths, res_vax = simulate_ring_vaccination(
    G, 
    ring_radius=0, 
    community_vax_coverage=0.75,
    community_vax_trigger=2, # Detection
    community_vax_delay=0.0,
    engine='python'
)
print(f"Python First Det Vax | Vaccines delivered: {res_vax} (Expected ~750)")
assert 730 <= res_vax <= 770, "Python First Detection vaccination count is off"


print("Testing C++ backend engine...")

# 3. Test Day 0 Vaccination (Trigger 1) with 50% coverage (CPP)
res_cases, res_deaths, res_vax = simulate_ring_vaccination(
    G, 
    ring_radius=0, 
    community_vax_coverage=0.5,
    community_vax_trigger=1, # Day 0
    community_vax_delay=0.0,
    engine='cpp'
)
print(f"CPP Day 0 Vax | Vaccines delivered: {res_vax} (Expected ~500)")
assert 480 <= res_vax <= 520, "CPP Day 0 vaccination count is off"


# 4. Test First Detection Vaccination (Trigger 2) with 75% coverage (CPP)
res_cases, res_deaths, res_vax = simulate_ring_vaccination(
    G, 
    ring_radius=0, 
    community_vax_coverage=0.75,
    community_vax_trigger=2, # Detection
    community_vax_delay=0.0,
    engine='cpp'
)
print(f"CPP First Det Vax | Vaccines delivered: {res_vax} (Expected ~750)")
assert 730 <= res_vax <= 770, "CPP First Detection vaccination count is off"

print("All tests passed!")
