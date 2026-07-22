import networkx as nx
import numpy as np
import heapq
import itertools
from scipy.stats import nbinom

def generate_network(N=10000, household_mean=5.0, community_mean=5.0, community_variance=160.0, household_size_dist=None):
    """
    Generates a Two-Layer Household-Community network to capture Ebola's high local clustering 
    (mutual contacts within households/caregiving settings) and overdispersed community spread (superspreading).
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # --- Layer 1: Household Cliques ---
    # Ebola spreads intensely within family/caregiving units.
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    idx = 0
    hh_dict = {}
    hh_counter = 0
    while idx < N:
        if isinstance(household_size_dist, str) and household_size_dist == 'poisson':
            hh_size = np.random.poisson(max(household_mean - 1.0, 0.1)) + 1
        else:
            if household_size_dist is None:
                # DRC DHS 2013-14 approximate household size distribution (1 to 10+)
                household_size_dist = [0.061, 0.097, 0.129, 0.147, 0.145, 0.128, 0.103, 0.076, 0.049, 0.065]
            # Normalize to ensure sum is 1.0
            household_size_dist = np.array(household_size_dist) / sum(household_size_dist)
            choices = np.arange(1, len(household_size_dist) + 1)
            hh_size = np.random.choice(choices, p=household_size_dist)
        if idx + hh_size > N:
            hh_size = N - idx
        hh_nodes = nodes[idx:idx+hh_size]
        for n in hh_nodes: hh_dict[n] = hh_counter
        
        # Wire households as fully connected cliques to create dense triangles (mutual contacts)
        for u, v in itertools.combinations(hh_nodes, 2):
            G.add_edge(u, v, weight=3.0)
        idx += hh_size
        hh_counter += 1
        
    # --- Layer 2: Community Spread (Negative Binomial) ---
    # Preserves the "superspreading" heavy-tail necessary for Ebola.
    mean_k = float(community_mean)
    var_k = float(community_variance)
    if var_k <= mean_k:
        var_k = mean_k + 1e-6
    
    p_nb = mean_k / var_k
    r_nb = mean_k**2 / (var_k - mean_k)
    
    ks = np.random.negative_binomial(r_nb, p_nb, N)
    if sum(ks) % 2 != 0:
        ks[0] += 1
        
    comm_G = nx.configuration_model(ks)
    comm_G = nx.Graph(comm_G) # Remove parallel edges
    comm_G.remove_edges_from(nx.selfloop_edges(comm_G))
    
    # Combine layers
    for u, v in comm_G.edges():
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0)
            
    nx.set_node_attributes(G, hh_dict, 'household')
    return G


def simulate_ring_vaccination(G, rt_array=None, baseline_tau=0.25, 
                              incubation_period=8.5, infectious_period=6.0,
                              ring_radius=1, efficacy=0.30, immune_delay=10.0, uptake=0.8,
                              reporting_rate=1.0, detection_delay=4.0, tracing_delay=2.0, max_cases=None,
                              max_daily_traces=100,
                              max_vaccines=None,
                              base_CFR=0.454, vax_CFR=None, vaccine_effect=None,
                              initial_infected=5, initial_exposed=0, max_sim_time=300, vax_start_time=0.0,
                              return_time_series=False, return_transmission_network=False,
                              risk_compensation_multiplier=1.0, trust_uptake_dependency=False,
                              tracing_coverage=None, vaccine_acceptability=None,
                              sigmoidal_k=0.5, sigmoidal_d0=None,
                              uptake_r2_drop=0.75, tracing_delay_r2_add=2.0, compete_queue=False, allow_pep=True, community_vax_coverage=0.0, community_vax_trigger=0, community_vax_delay=-1.0, community_vax_rollout_days=0.0, seed=None, engine='cpp', return_mechanism=False,
                              infection_efficacy_multiplier=1.0, immediate_mortality_protection=False, incubation_shape=1.0, infectious_shape=1.0):
    """
    Simulates an SEIR Ebola outbreak with a targeted ring vaccination intervention, incorporating logistical delays.
    """
    if sigmoidal_d0 is None:
        sigmoidal_d0 = immune_delay
    explicit_vax_CFR = vax_CFR
    if vaccine_effect is not None:
        efficacy = float(vaccine_effect)
        if explicit_vax_CFR is None:
            vax_CFR = float(base_CFR) * (1.0 - efficacy)
    elif explicit_vax_CFR is None:
        vax_CFR = float(base_CFR) * (1.0 - efficacy)

    effective_vaccine_acceptability = vaccine_acceptability
    if tracing_coverage is not None and effective_vaccine_acceptability is None:
        effective_vaccine_acceptability = 1.0

    if engine in ['cpp', 'cohort']:
        import ebola_stochastic_ring_cpp
        N = G.number_of_nodes()
        if not hasattr(G, 'cpp_engine'):
            offsets = np.zeros(N + 1, dtype=np.int32)
            edges = []
            for i in range(N):
                offsets[i] = len(edges)
                edges.extend([int(x) for x in G.neighbors(i)])
            offsets[N] = len(edges)
            edges = np.array(edges, dtype=np.int32)
            G.cpp_engine = ebola_stochastic_ring_cpp.EbolaEngine(N, offsets, edges)
        
        # Format reporting rate array
        rr_array = []
        rr_scalar = 1.0
        if isinstance(reporting_rate, list):
            rr_array = reporting_rate
        else:
            rr_scalar = reporting_rate
            
        res = G.cpp_engine.run_simulation(
            [float(x) for x in rt_array] if rt_array is not None else [], float(baseline_tau),
            float(incubation_period), float(infectious_period),
            int(ring_radius), float(efficacy), float(immune_delay), float(uptake),
            [float(x) for x in rr_array] if rr_array else [], float(rr_scalar),
            float(detection_delay), float(tracing_delay), int(max_cases if max_cases is not None else -1),
            int(max_daily_traces), int(max_vaccines if max_vaccines is not None else -1),
            float(base_CFR), float(vax_CFR), int(initial_infected), int(initial_exposed),
            int(max_sim_time), float(vax_start_time), bool(return_time_series),
            float(risk_compensation_multiplier), bool(trust_uptake_dependency),
            [float(x) for x in tracing_coverage] if isinstance(tracing_coverage, (list, np.ndarray)) else ([] if tracing_coverage is None else [float(tracing_coverage)]), 
            float(effective_vaccine_acceptability if effective_vaccine_acceptability is not None else -1.0),
            float(sigmoidal_k if sigmoidal_k is not None else -1.0), float(sigmoidal_d0 if sigmoidal_d0 is not None else -1.0),
            float(uptake_r2_drop), float(tracing_delay_r2_add), bool(compete_queue), bool(allow_pep), float(community_vax_coverage), int(community_vax_trigger), float(community_vax_delay), float(community_vax_rollout_days), int(seed if seed is not None else -1),
            float(infection_efficacy_multiplier), bool(immediate_mortality_protection),
            bool(return_mechanism), float(incubation_shape), float(infectious_shape),
            bool(engine in ['cpp', 'cohort'])
        )
        if return_mechanism:
            return res
        if return_time_series:
            return res
        return res[0], res[1], res[2], res[3]

    N = G.number_of_nodes()
    gamma = 1.0 / infectious_period
    sigma = 1.0 / incubation_period
    
    def get_rt(t):
        if rt_array is not None:
            idx = int(t)
            if idx < len(rt_array):
                return rt_array[idx]
            else:
                return rt_array[-1]
        return 1.66
        
    R_max = max(rt_array) if rt_array is not None else 1.66
    R_max = max(R_max, 0.01)
    
    tau_max = baseline_tau
    
    def scale_func(t):
        return get_rt(t) / R_max
        
    def get_trace_probability(distance, t=0.0):
        base_up = uptake if distance == 1 else (uptake * uptake_r2_drop)
        if tracing_coverage is not None and effective_vaccine_acceptability is not None:
            if isinstance(tracing_coverage, (list, np.ndarray)):
                idx = min(int(t), len(tracing_coverage) - 1)
                base_up = tracing_coverage[idx]
            else:
                base_up = tracing_coverage
        return base_up

    def get_vaccine_acceptance(distance):
        base_up = uptake if distance == 1 else (uptake * uptake_r2_drop)
        if tracing_coverage is not None and effective_vaccine_acceptability is not None:
            base_up = effective_vaccine_acceptability
        if trust_uptake_dependency:
            base_up = base_up * efficacy
        return base_up
        
    def get_node_delay(distance):
        return tracing_delay if distance == 1 else (tracing_delay + tracing_delay_r2_add)
    
    # Node states: 'S', 'E', 'I', 'R', 'V'
    status = {n: 'S' for n in G.nodes()}
    received_vaccine = {n: False for n in G.nodes()} # Tracks if they ever got the shot, for therapeutic CFR benefit
    monitored = {n: False for n in G.nodes()}
    vaccination_time = {}
    exposed_before_vaccination = {n: False for n in G.nodes()}
    scheduled_vaccine_nodes = set()
    
    def get_efficacy(target_node, current_time):
        if not received_vaccine[target_node]: return 0.0
        d = current_time - vaccination_time.get(target_node, current_time)
        if d < 0: return 0.0
        if sigmoidal_k is not None and sigmoidal_d0 is not None:
            return infection_efficacy_multiplier * efficacy / (1.0 + np.exp(-sigmoidal_k * (d - sigmoidal_d0)))
        else:
            return infection_efficacy_multiplier * efficacy if d >= immune_delay else 0.0
            
    def get_therapeutic_efficacy(target_node, current_time):
        if not received_vaccine[target_node]: return 0.0
        if immediate_mortality_protection: return 1.0
        d = current_time - vaccination_time.get(target_node, current_time)
        if d < 0: return 0.0
        if sigmoidal_k is not None and sigmoidal_d0 is not None:
            # Therapeutic benefit scales from 0 to 1.0 (100% of the CFR rescue)
            return 1.0 / (1.0 + np.exp(-sigmoidal_k * (d - sigmoidal_d0)))
        else:
            return 1.0 if d >= immune_delay else 0.0
    
    queue = []
    counter = itertools.count()
    
    def add_event(t, event_type, target, source=None, rec_time=None):
        heapq.heappush(queue, (t, next(counter), event_type, target, source, rec_time))
        
    initial_count = min(N, int(initial_infected) + int(initial_exposed))
    initial_nodes_all = np.random.choice(G.nodes(), size=initial_count, replace=False)
    initial_nodes = initial_nodes_all[:int(initial_infected)]
    initial_exposed_nodes = initial_nodes_all[int(initial_infected):]
    total_infected = 0
    total_deaths = 0
    total_vaccines = 0
    next_available_trace_time = 0.0
    first_detection_occurred = False
    community_vax_order = []
    community_vax_next_index = 0
    community_vax_daily_quota = 0
    
    daily_incidence = np.zeros(max_sim_time + 1)
    transmission_edges = []
    
    def schedule_community_vax(start_t):
        nonlocal community_vax_order, community_vax_next_index, community_vax_daily_quota
        if community_vax_coverage <= 0.0:
            return
        num_to_vax = int(np.round(N * community_vax_coverage))
        if num_to_vax <= 0:
            return
        community_vax_order = list(G.nodes())
        np.random.shuffle(community_vax_order)
        community_vax_next_index = 0
        if community_vax_rollout_days > 0.0:
            community_vax_daily_quota = max(1, int(np.ceil(num_to_vax / community_vax_rollout_days)))
        else:
            community_vax_daily_quota = num_to_vax
        add_event(start_t, 'COMMUNITY_VAX', -1)

    if community_vax_trigger == 1:
        schedule_community_vax(community_vax_delay)
        
    for n in initial_nodes:
        add_event(0.0, 'EXPOSURE', n)
    for n in initial_exposed_nodes:
        status[n] = 'E'
        exposed_before_vaccination[n] = True
        total_infected += 1
        add_event(np.random.exponential(1.0 / sigma), 'ONSET', n)
        
    while queue:
        t, _, event_type, target, source, rec_time = heapq.heappop(queue)
        
        if t > max_sim_time:
            break
            
        if event_type == 'EXPOSURE':
            if status[target] in ['E', 'I', 'R']:
                continue
                
            if np.random.rand() < get_efficacy(target, t):
                continue # Vaccine protected them
                    
            status[target] = 'E'
            exposed_before_vaccination[target] = not received_vaccine[target]
            if source is not None:
                try:
                    weight = G.edges[source, target].get('weight', 1.0)
                except:
                    weight = 1.0
                transmission_edges.append((source, target, weight, t))
            total_infected += 1
            
            if max_cases and total_infected >= max_cases:
                if return_time_series: return daily_incidence
                return total_infected / N, total_deaths / N, total_vaccines
            
            onset_time = t + np.random.exponential(1.0 / sigma)
            add_event(onset_time, 'ONSET', target)
            
        elif event_type == 'EXPOSURE_CANDIDATE':
            if status[target] in ['E', 'I', 'R', 'ISO']:
                continue
                
            # If the source infector was isolated before this exposure could occur, the transmission is blocked.
            if source is not None and status.get(source) == 'ISO':
                continue
                
            prob_accept = scale_func(t)
            accepted = (np.random.rand() < prob_accept)
            
            if accepted:
                is_protected = False
                if np.random.rand() < get_efficacy(target, t):
                    is_protected = True
                    
                if not is_protected:
                    status[target] = 'E'
                    exposed_before_vaccination[target] = not received_vaccine[target]
                    if source is not None:
                        try:
                            weight = G.edges[source, target].get('weight', 1.0)
                        except:
                            weight = 1.0
                        transmission_edges.append((source, target, weight, t))
                    total_infected += 1
                    if max_cases and total_infected >= max_cases:
                        if return_time_series: return daily_incidence
                        return total_infected / N, total_deaths / N, total_vaccines
                    onset_time = t + np.random.exponential(1.0 / sigma)
                    add_event(onset_time, 'ONSET', target)
                    continue 
                    
            eff_tau = tau_max
            if status[target] == 'V' and risk_compensation_multiplier > 1.0:
                eff_tau *= risk_compensation_multiplier
                
            next_t = t + np.random.exponential(1.0 / eff_tau)
            if next_t < rec_time:
                add_event(next_t, 'EXPOSURE_CANDIDATE', target, source, rec_time)
                
        elif event_type == 'ONSET':
            if status[target] == 'E':
                pep_eligible = allow_pep or not exposed_before_vaccination[target]
                if pep_eligible and received_vaccine[target] and np.random.rand() < get_efficacy(target, t):
                    status[target] = 'R'
                    total_infected -= 1
                    continue

                status[target] = 'I'
                
                if int(t) <= max_sim_time:
                    daily_incidence[int(t)] += 1
                    
                # Check for mortality based on continuous sigmoidal therapeutic rescue
                therapeutic_benefit = get_therapeutic_efficacy(target, t)
                cfr = base_CFR - therapeutic_benefit * (base_CFR - vax_CFR)
                
                if np.random.rand() < cfr:
                    total_deaths += 1
                
                rec_t = t + np.random.exponential(1.0 / gamma)
                add_event(rec_t, 'RECOVERY', target)
                
                # Detection triggers the ring vaccination process if the case is reported
                current_rr = reporting_rate
                if isinstance(reporting_rate, list):
                    idx = min(int(t), len(reporting_rate) - 1)
                    current_rr = reporting_rate[idx]
                    
                if np.random.rand() < current_rr:
                    det_time = t + (1.0 if monitored[target] else detection_delay)
                    if det_time < rec_t:
                        add_event(det_time, 'DETECTION', target)
                    
                for neighbor in G.neighbors(target):
                    if status[neighbor] in ['S', 'V']:
                        eff_tau = tau_max
                        if status[neighbor] == 'V' and risk_compensation_multiplier > 1.0:
                            eff_tau *= risk_compensation_multiplier
                            
                        inf_time = t + np.random.exponential(1.0 / eff_tau)
                        if inf_time < rec_t:
                            add_event(inf_time, 'EXPOSURE_CANDIDATE', neighbor, target, rec_t)
                            
        elif event_type == 'RECOVERY':
            if status[target] in ['I', 'ISO']:
                status[target] = 'R'
                
        elif event_type == 'DETECTION':
            if t > max_sim_time:
                continue
                
            if not first_detection_occurred and community_vax_trigger == 2:
                first_detection_occurred = True
                schedule_community_vax(t + community_vax_delay)
                
            if t < vax_start_time:
                continue

            # Index case is isolated upon detection, stopping further outgoing transmissions
            if status[target] == 'I':
                status[target] = 'ISO'
                
            ring_nodes = nx.single_source_shortest_path_length(G, target, cutoff=ring_radius)
            for neighbor, distance in ring_nodes.items():
                if distance == 0: continue
                # Monitoring is independent of vaccination status. Vaccinated
                # contacts can still become cases and should receive shortened
                # detection delays if they are traced.
                if status[neighbor] in ['S', 'E', 'V']:
                    if np.random.rand() < get_trace_probability(distance, t):
                        monitored[neighbor] = True
                    else:
                        continue

                    if neighbor in scheduled_vaccine_nodes or received_vaccine[neighbor]:
                        continue

                    if np.random.rand() < get_vaccine_acceptance(distance):
                        if max_vaccines is not None and len(scheduled_vaccine_nodes) + total_vaccines >= max_vaccines:
                            continue
                        node_delay = get_node_delay(distance)
                        # Queueing logic: contact tracers have a daily bandwidth
                        trace_start_t = max(t, next_available_trace_time)
                        vax_t = trace_start_t + node_delay
                        next_available_trace_time = trace_start_t + (1.0 / max_daily_traces)
                        
                        scheduled_vaccine_nodes.add(neighbor)
                        add_event(vax_t, 'VACCINATION', neighbor)
                        
        elif event_type == 'VACCINATION':
            if t > max_sim_time:
                continue
            scheduled_vaccine_nodes.discard(target)
            if status[target] in ['S', 'E']: # E nodes can still receive shot for CFR benefit, but status remains E
                if not received_vaccine[target]:
                    total_vaccines += 1
                received_vaccine[target] = True
                vaccination_time[target] = t # Always set the time they got the shot, even if already E
            
            if status[target] == 'S':
                status[target] = 'V'
                
        elif event_type == 'COMMUNITY_VAX':
            if t > max_sim_time:
                continue
            num_to_vax = int(np.round(N * community_vax_coverage))
            stop_index = min(num_to_vax, community_vax_next_index + community_vax_daily_quota)
            for i in range(community_vax_next_index, stop_index):
                curr = community_vax_order[i]
                if status[curr] in ['S', 'E']:
                    if not received_vaccine[curr]:
                        scheduled_vaccine_nodes.discard(curr)
                        total_vaccines += 1
                        received_vaccine[curr] = True
                        vaccination_time[curr] = t
                        if status[curr] == 'S':
                            status[curr] = 'V'
            community_vax_next_index = stop_index
            if community_vax_next_index < num_to_vax and community_vax_rollout_days > 0.0:
                add_event(t + 1.0, 'COMMUNITY_VAX', -1)
                        
    if return_transmission_network:
        return transmission_edges
    if return_time_series:
        return daily_incidence
    
    last_event_time = 0.0
    for t_val, _, _, _, _, _ in queue_history:
        if t_val <= max_sim_time:
            last_event_time = max(last_event_time, t_val)
            
    return total_infected / N, total_deaths / N, total_vaccines, last_event_time

if __name__ == "__main__":
    G = generate_network(5000)
    print("Clustering:", nx.average_clustering(G))
    tau = calibrate_tau(G, 1.6, 1.0/6.0)
    print("Calibrated baseline tau:", tau)
    res, deaths, vaccines = simulate_ring_vaccination(G, rt_array=None, baseline_tau=tau, incubation_period=8.5, infectious_period=6.0, uptake=0.5, efficacy=0.3, reporting_rate=0.5)
    print(f"Final outbreak size: {res*100:.2f}% (Cases), {deaths*100:.2f}% (Deaths), {vaccines} Vaccines used")
