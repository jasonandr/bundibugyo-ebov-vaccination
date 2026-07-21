#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace py = pybind11;

struct Event {
    double t;
    long long id;
    int type; // 0: EXPOSURE, 1: EXPOSURE_CANDIDATE, 2: ONSET, 3: DETECTION, 4: VACCINATION, 5: RECOVERY
    int target;
    int source;
    double rec_time;

    bool operator<(const Event& other) const {
        if (t != other.t) return t > other.t; // Min-heap
        return id > other.id;
    }
};

py::object simulate_ring_vaccination_cpp(
    int N,
    const std::vector<std::vector<int>>& adj,
    const std::vector<double>& rt_array,
    double baseline_tau,
    double incubation_period,
    double infectious_period,
    int ring_radius,
    double efficacy,
    double immune_delay,
    double uptake,
    const std::vector<double>& reporting_rate,
    double reporting_rate_scalar,
    double detection_delay,
    double tracing_delay,
    int max_cases,
    int max_daily_traces,
    int max_vaccines,
    double base_CFR,
    double vax_CFR,
    int initial_infected,
    int initial_exposed,
    int max_sim_time,
    double vax_start_time,
    bool return_time_series,
    double risk_compensation_multiplier,
    bool trust_uptake_dependency,
    const std::vector<double>& tracing_coverage,
    double vaccine_acceptability,
    double sigmoidal_k,
    double sigmoidal_d0,
    double uptake_r2_drop,
    double tracing_delay_r2_add,
    bool compete_queue,
    bool allow_pep,
    double community_vax_coverage,
    int community_vax_trigger,
    double community_vax_delay,
    double community_vax_rollout_days,
    int seed,
    double infection_efficacy_multiplier,
    bool immediate_mortality_protection
) {
    std::mt19937 gen(seed < 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<> runif(0.0, 1.0);
    std::exponential_distribution<> rexp_gamma(1.0 / infectious_period);
    std::exponential_distribution<> rexp_sigma(1.0 / incubation_period);

    double R_max = rt_array.empty() ? 1.66 : 0.0;
    if (!rt_array.empty()) {
        for (double v : rt_array) if (v > R_max) R_max = v;
    }
    R_max = std::max(R_max, 0.01);

    auto get_rt = [&](double t) {
        if (!rt_array.empty()) {
            int idx = static_cast<int>(t);
            if (idx < rt_array.size()) return rt_array[idx];
            return rt_array.back();
        }
        return 1.66;
    };

    auto scale_func = [&](double t) {
        return get_rt(t) / R_max;
    };

    auto get_trace_probability = [&](int distance, double t) {
        double base_up = (distance == 1) ? uptake : (uptake * uptake_r2_drop);
        if (tracing_coverage.size() > 0 && vaccine_acceptability >= 0) {
            int idx = std::min((int)t, (int)tracing_coverage.size() - 1);
            base_up = tracing_coverage[idx];
        }
        return base_up;
    };

    auto get_vaccine_acceptance = [&](int distance) {
        double base_up = (distance == 1) ? uptake : (uptake * uptake_r2_drop);
        if (tracing_coverage.size() > 0 && vaccine_acceptability >= 0) {
            base_up = vaccine_acceptability;
        }
        if (trust_uptake_dependency) {
            base_up *= efficacy;
        }
        return base_up;
    };

    auto get_node_delay = [&](int distance) {
        return (distance == 1) ? tracing_delay : (tracing_delay + tracing_delay_r2_add);
    };

    std::vector<int> status(N, 0); // 0: S, 1: E, 2: I, 3: R, 4: V, 5: ISO
    std::vector<bool> received_vaccine(N, false);
    std::vector<bool> monitored(N, false);
    std::vector<double> vaccination_time(N, -1.0);
    std::vector<bool> exposed_before_vaccination(N, false);
    std::unordered_set<int> scheduled_vaccine_nodes;

    auto get_efficacy = [&](int target_node, double current_time) {
        if (!received_vaccine[target_node]) return 0.0;
        double d = current_time - vaccination_time[target_node];
        if (d < 0) return 0.0;
        double eff = 0.0;
        if (sigmoidal_k >= 0 && sigmoidal_d0 >= 0) {
            eff = efficacy / (1.0 + std::exp(-sigmoidal_k * (d - sigmoidal_d0)));
        } else {
            eff = (d >= immune_delay) ? efficacy : 0.0;
        }
        return eff * infection_efficacy_multiplier;
    };

    auto get_therapeutic_efficacy = [&](int target_node, double current_time) {
        if (!received_vaccine[target_node]) return 0.0;
        if (immediate_mortality_protection) return 1.0;
        double d = current_time - vaccination_time[target_node];
        if (d < 0) return 0.0;
        if (sigmoidal_k >= 0 && sigmoidal_d0 >= 0) {
            return 1.0 / (1.0 + std::exp(-sigmoidal_k * (d - sigmoidal_d0)));
        }
        return (d >= immune_delay) ? 1.0 : 0.0;
    };

    std::priority_queue<Event> pq;
    long long counter = 0;

    auto add_event = [&](double t, int type, int target, int source = -1, double rec_time = -1.0) {
        pq.push({t, counter++, type, target, source, rec_time});
    };

    std::vector<int> initial_nodes;
    std::vector<int> nodes(N);
    for (int i = 0; i < N; ++i) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), gen);
    int initial_count = std::min(N, initial_infected + initial_exposed);
    for (int i = 0; i < initial_infected && i < N; ++i) {
        initial_nodes.push_back(nodes[i]);
    }
    std::vector<int> initial_exposed_nodes;
    for (int i = initial_infected; i < initial_count; ++i) {
        initial_exposed_nodes.push_back(nodes[i]);
    }

    int total_infected = 0;
    int total_deaths = 0;
    int total_vaccines = 0;
    double next_available_trace_time = 0.0;
    double tau_max = baseline_tau;
    
    std::vector<int> daily_deaths(max_sim_time + 2, 0);

    std::vector<int> community_vax_order;
    int community_vax_next_index = 0;
    double community_vax_daily_quota = 0.0;

    auto schedule_community_vax = [&](double start_t) {
        if (community_vax_coverage <= 0.0) return;
        int num_to_vax = std::round(N * community_vax_coverage);
        if (num_to_vax <= 0) return;
        community_vax_order.resize(N);
        std::iota(community_vax_order.begin(), community_vax_order.end(), 0);
        std::shuffle(community_vax_order.begin(), community_vax_order.end(), gen);
        community_vax_next_index = 0;
        if (community_vax_rollout_days > 0.0) {
            community_vax_daily_quota = std::max(1.0, std::ceil(num_to_vax / community_vax_rollout_days));
        } else {
            community_vax_daily_quota = num_to_vax;
        }
        add_event(start_t, 6, -1); // 6: COMMUNITY_VAX
    };

    bool first_detection_occurred = false;
    if (community_vax_trigger == 1) {
        schedule_community_vax(community_vax_delay);
    }

    for (int n : initial_nodes) {
        add_event(-detection_delay, 2, n); // 2: ONSET
    }
    for (int n : initial_exposed_nodes) {
        status[n] = 1; // E
        exposed_before_vaccination[n] = true;
        total_infected++;
        add_event(rexp_sigma(gen), 2, n); // residual time to symptom onset
    }

    while (!pq.empty()) {
        Event ev = pq.top();
        pq.pop();

        double t = ev.t;

        int target = ev.target;
        int source = ev.source;

        if (ev.type == 0) { // EXPOSURE
            if (t > max_sim_time) continue;
            if (status[target] == 1 || status[target] == 2 || status[target] == 3) continue; // E, I, R
            if (runif(gen) < get_efficacy(target, t)) continue;
            
            if (!received_vaccine[target]) {
                exposed_before_vaccination[target] = true;
            }
            status[target] = 1; // E
            total_infected++;
            if (max_cases > 0 && total_infected >= max_cases) continue;

            add_event(t + rexp_sigma(gen), 2, target); // 2: ONSET
        } 
        else if (ev.type == 1) { // EXPOSURE_CANDIDATE
            if (t > max_sim_time) continue;
            if (status[target] == 1 || status[target] == 2 || status[target] == 3 || status[target] == 5) continue;
            if (source != -1 && status[source] == 5) continue; // ISO
            
            if (runif(gen) < scale_func(t)) {
                if (runif(gen) >= get_efficacy(target, t)) {
                    if (!received_vaccine[target]) {
                        exposed_before_vaccination[target] = true;
                    }
                    status[target] = 1; // E
                    total_infected++;
                    if (max_cases > 0 && total_infected >= max_cases) continue;
                    add_event(t + rexp_sigma(gen), 2, target);
                    continue;
                }
            }
            
            double eff_tau = tau_max;
            if (status[target] == 4 && risk_compensation_multiplier > 1.0) {
                eff_tau *= risk_compensation_multiplier;
            }
            std::exponential_distribution<> rexp_tau(eff_tau);
            double next_t = t + rexp_tau(gen);
            if (next_t < ev.rec_time) {
                add_event(next_t, 1, target, source, ev.rec_time);
            }
        }
        else if (ev.type == 2) { // ONSET
            if (status[target] == 1 || status[target] == 0) {
                if (status[target] == 1 && received_vaccine[target]) {
                    bool eligible = true;
                    if (!allow_pep && exposed_before_vaccination[target]) {
                        eligible = false;
                    }
                    if (eligible && runif(gen) < get_efficacy(target, t)) {
                        status[target] = 3; // R
                        total_infected--; // Aborted clinical infection
                        continue;
                    }
                }
                
                if (status[target] == 0) total_infected++;
                status[target] = 2; // I
                double tb = get_therapeutic_efficacy(target, t);
                double cfr = base_CFR - tb * (base_CFR - vax_CFR);
                if (runif(gen) < cfr) {
                    total_deaths++;
                    if (return_time_series && t >= 0 && t <= max_sim_time) {
                        daily_deaths[int(t)]++;
                    }
                }

                double rec_t = t + rexp_gamma(gen);
                add_event(rec_t, 5, target); // 5: RECOVERY
                
                double current_rr = reporting_rate_scalar;
                if (!reporting_rate.empty()) {
                    int idx = std::min((int)t, (int)reporting_rate.size() - 1);
                    current_rr = reporting_rate[idx];
                }
                
                if (runif(gen) < current_rr) {
                    double det_time = t + (monitored[target] ? 1.0 : detection_delay);
                    if (det_time < rec_t) {
                        add_event(det_time, 3, target); // 3: DETECTION
                    }
                }
                
                for (int neighbor : adj[target]) {
                    if (status[neighbor] == 0 || status[neighbor] == 4) { // S or V
                        double eff_tau = tau_max;
                        if (status[neighbor] == 4 && risk_compensation_multiplier > 1.0) {
                            eff_tau *= risk_compensation_multiplier;
                        }
                        std::exponential_distribution<> rexp_tau(eff_tau);
                        double inf_time = t + rexp_tau(gen);
                        if (inf_time < rec_t) {
                            add_event(inf_time, 1, neighbor, target, rec_t);
                        }
                    }
                }
            }
        }
        else if (ev.type == 5) { // RECOVERY
            if (status[target] == 2 || status[target] == 5) {
                status[target] = 3; // R
            }
        }
        else if (ev.type == 3) { // DETECTION
            if (t > max_sim_time) continue;
            
            if (!first_detection_occurred && community_vax_trigger == 2) {
                first_detection_occurred = true;
                schedule_community_vax(t + community_vax_delay);
            }
            
            if (t < vax_start_time) continue;
            if (status[target] == 2) status[target] = 5; // ISO

            // BFS for ring vaccination
            std::vector<int> distances(N, -1);
            std::queue<int> q;
            q.push(target);
            distances[target] = 0;
            
            std::vector<int> candidates;

            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                int d = distances[curr];
                if (d > 0) {
                    if (status[curr] == 0 || status[curr] == 1 || status[curr] == 4) { // S, E, or vaccinated susceptible
                        if (runif(gen) < get_trace_probability(d, t)) {
                            monitored[curr] = true;
                            if (scheduled_vaccine_nodes.find(curr) == scheduled_vaccine_nodes.end() && !received_vaccine[curr]) {
                                candidates.push_back(curr);
                            }
                        }
                    }
                }
                
                if (d < ring_radius) {
                    for (int neighbor : adj[curr]) {
                        if (distances[neighbor] == -1) {
                            distances[neighbor] = d + 1;
                            q.push(neighbor);
                        }
                    }
                }
            }

            if (compete_queue) {
                std::shuffle(candidates.begin(), candidates.end(), gen);
            }

            for (int curr : candidates) {
                int d = distances[curr];
                if (runif(gen) >= get_vaccine_acceptance(d)) continue;
                if (max_vaccines >= 0 && scheduled_vaccine_nodes.size() + total_vaccines >= max_vaccines) break;
                double trace_start_t = std::max(t, next_available_trace_time);
                double vax_t = trace_start_t + get_node_delay(d);
                next_available_trace_time = trace_start_t + (1.0 / max_daily_traces);
                
                scheduled_vaccine_nodes.insert(curr);
                add_event(vax_t, 4, curr); // 4: VACCINATION
            }
        }
        else if (ev.type == 4) { // VACCINATION
            if (t > max_sim_time) continue;
            scheduled_vaccine_nodes.erase(target);
            if (status[target] == 0 || status[target] == 1) {
                if (!received_vaccine[target]) total_vaccines++;
                received_vaccine[target] = true;
                vaccination_time[target] = t;
            }
            if (status[target] == 0) status[target] = 4; // V
        }
        else if (ev.type == 6) { // COMMUNITY_VAX
            if (t > max_sim_time) continue;
            int num_to_vax = std::round(N * community_vax_coverage);
            int stop_index = std::min(num_to_vax, community_vax_next_index + (int)community_vax_daily_quota);
            for (int i = community_vax_next_index; i < stop_index; ++i) {
                int curr = community_vax_order[i];
                if (status[curr] == 0 || status[curr] == 1) {
                    if (!received_vaccine[curr]) {
                        scheduled_vaccine_nodes.erase(curr);
                        total_vaccines++;
                        received_vaccine[curr] = true;
                        vaccination_time[curr] = t;
                        if (status[curr] == 0) status[curr] = 4; // V
                    }
                }
            }
            community_vax_next_index = stop_index;
            if (community_vax_next_index < num_to_vax && community_vax_rollout_days > 0.0) {
                add_event(t + 1.0, 6, -1);
            }
        }
    }

    if (return_time_series) {
        return py::cast(daily_deaths);
    }
    return py::make_tuple((double)total_infected / N, (double)total_deaths / N, total_vaccines);
}

py::object simulate_mechanism_cpp(
    int N,
    const std::vector<std::vector<int>>& adj,
    const std::vector<double>& rt_array,
    double baseline_tau,
    double incubation_period,
    double infectious_period,
    int ring_radius,
    double efficacy,
    double immune_delay,
    double uptake,
    const std::vector<double>& reporting_rate,
    double reporting_rate_scalar,
    double detection_delay,
    double tracing_delay,
    int max_cases,
    int max_daily_traces,
    int max_vaccines,
    double base_CFR,
    double vax_CFR,
    int initial_infected,
    int initial_exposed,
    int max_sim_time,
    double vax_start_time,
    bool return_time_series,
    double risk_compensation_multiplier,
    bool trust_uptake_dependency,
    const std::vector<double>& tracing_coverage,
    double vaccine_acceptability,
    double sigmoidal_k,
    double sigmoidal_d0,
    double uptake_r2_drop,
    double tracing_delay_r2_add,
    bool compete_queue,
    bool allow_pep,
    double community_vax_coverage,
    int community_vax_trigger,
    double community_vax_delay,
    double community_vax_rollout_days,
    int seed,
    double infection_efficacy_multiplier,
    bool immediate_mortality_protection
) {
    std::mt19937 gen(seed < 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<> runif(0.0, 1.0);
    std::exponential_distribution<> rexp_gamma(1.0 / infectious_period);
    std::exponential_distribution<> rexp_sigma(1.0 / incubation_period);

    double R_max = rt_array.empty() ? 1.66 : 0.0;
    if (!rt_array.empty()) {
        for (double v : rt_array) if (v > R_max) R_max = v;
    }
    R_max = std::max(R_max, 0.01);

    auto get_rt = [&](double t) {
        if (!rt_array.empty()) {
            int idx = static_cast<int>(t);
            if (idx < rt_array.size()) return rt_array[idx];
            return rt_array.back();
        }
        return 1.66;
    };

    auto scale_func = [&](double t) {
        return get_rt(t) / R_max;
    };

    auto get_trace_probability = [&](int distance, double t) {
        double base_up = (distance == 1) ? uptake : (uptake * uptake_r2_drop);
        if (tracing_coverage.size() > 0 && vaccine_acceptability >= 0) {
            int idx = std::min((int)t, (int)tracing_coverage.size() - 1);
            base_up = tracing_coverage[idx];
        }
        return base_up;
    };

    auto get_vaccine_acceptance = [&](int distance) {
        double base_up = (distance == 1) ? uptake : (uptake * uptake_r2_drop);
        if (tracing_coverage.size() > 0 && vaccine_acceptability >= 0) {
            base_up = vaccine_acceptability;
        }
        if (trust_uptake_dependency) {
            base_up *= efficacy;
        }
        return base_up;
    };

    auto get_node_delay = [&](int distance) {
        return (distance == 1) ? tracing_delay : (tracing_delay + tracing_delay_r2_add);
    };

    std::vector<int> status(N, 0);

    std::vector<double> exposure_time(N, -1.0);
    std::vector<double> onset_time(N, -1.0);
    std::vector<double> recovery_or_death_time(N, -1.0);
    std::vector<bool> died(N, false);
    std::vector<bool> aborted_due_to_pep(N, false);
    std::vector<int> state_at_vaccination(N, -1);

    std::vector<bool> received_vaccine(N, false);
    std::vector<bool> monitored(N, false);
    std::vector<double> vaccination_time(N, -1.0);
    std::vector<bool> exposed_before_vaccination(N, false);
    std::unordered_set<int> scheduled_vaccine_nodes;

    auto get_efficacy = [&](int target_node, double current_time) {
        if (!received_vaccine[target_node]) return 0.0;
        double d = current_time - vaccination_time[target_node];
        if (d < 0) return 0.0;
        double eff = 0.0;
        if (sigmoidal_k >= 0 && sigmoidal_d0 >= 0) {
            eff = efficacy / (1.0 + std::exp(-sigmoidal_k * (d - sigmoidal_d0)));
        } else {
            eff = (d >= immune_delay) ? efficacy : 0.0;
        }
        return eff * infection_efficacy_multiplier;
    };

    auto get_therapeutic_efficacy = [&](int target_node, double current_time) {
        if (!received_vaccine[target_node]) return 0.0;
        if (immediate_mortality_protection) return 1.0;
        double d = current_time - vaccination_time[target_node];
        if (d < 0) return 0.0;
        if (sigmoidal_k >= 0 && sigmoidal_d0 >= 0) {
            return 1.0 / (1.0 + std::exp(-sigmoidal_k * (d - sigmoidal_d0)));
        }
        return (d >= immune_delay) ? 1.0 : 0.0;
    };

    std::priority_queue<Event> pq;
    long long counter = 0;

    auto add_event = [&](double t, int type, int target, int source = -1, double rec_time = -1.0) {
        pq.push({t, counter++, type, target, source, rec_time});
    };

    std::vector<int> initial_nodes;
    std::vector<int> nodes(N);
    for (int i = 0; i < N; ++i) nodes[i] = i;
    std::shuffle(nodes.begin(), nodes.end(), gen);
    int initial_count = std::min(N, initial_infected + initial_exposed);
    for (int i = 0; i < initial_infected && i < N; ++i) {
        initial_nodes.push_back(nodes[i]);
    }
    std::vector<int> initial_exposed_nodes;
    for (int i = initial_infected; i < initial_count; ++i) {
        initial_exposed_nodes.push_back(nodes[i]);
    }

    int total_infected = 0;
    int total_deaths = 0;
    int total_vaccines = 0;
    double next_available_trace_time = 0.0;
    double tau_max = baseline_tau;
    
    std::vector<int> daily_deaths(max_sim_time + 2, 0);

    std::vector<int> community_vax_order;
    int community_vax_next_index = 0;
    double community_vax_daily_quota = 0.0;

    auto schedule_community_vax = [&](double start_t) {
        if (community_vax_coverage <= 0.0) return;
        int num_to_vax = std::round(N * community_vax_coverage);
        if (num_to_vax <= 0) return;
        community_vax_order.resize(N);
        std::iota(community_vax_order.begin(), community_vax_order.end(), 0);
        std::shuffle(community_vax_order.begin(), community_vax_order.end(), gen);
        community_vax_next_index = 0;
        if (community_vax_rollout_days > 0.0) {
            community_vax_daily_quota = std::max(1.0, std::ceil(num_to_vax / community_vax_rollout_days));
        } else {
            community_vax_daily_quota = num_to_vax;
        }
        add_event(start_t, 6, -1); // 6: COMMUNITY_VAX
    };

    bool first_detection_occurred = false;
    if (community_vax_trigger == 1) {
        schedule_community_vax(community_vax_delay);
    }

    for (int n : initial_nodes) {
        add_event(-detection_delay, 2, n); // 2: ONSET
    }
    for (int n : initial_exposed_nodes) {
        status[n] = 1; // E
        exposed_before_vaccination[n] = true;
        total_infected++;
        add_event(rexp_sigma(gen), 2, n); // residual time to symptom onset
    }

    while (!pq.empty()) {
        Event ev = pq.top();
        pq.pop();

        double t = ev.t;

        int target = ev.target;
        int source = ev.source;

        if (ev.type == 0) { // EXPOSURE
            if (t > max_sim_time) continue;
            if (status[target] == 1 || status[target] == 2 || status[target] == 3) continue; // E, I, R
            if (runif(gen) < get_efficacy(target, t)) continue;
            
            if (!received_vaccine[target]) {
                exposed_before_vaccination[target] = true;
            }
            status[target] = 1;
            exposure_time[target] = t;
            total_infected++;
            if (max_cases > 0 && total_infected >= max_cases) continue;

            add_event(t + rexp_sigma(gen), 2, target); // 2: ONSET
        } 
        else if (ev.type == 1) { // EXPOSURE_CANDIDATE
            if (t > max_sim_time) continue;
            if (status[target] == 1 || status[target] == 2 || status[target] == 3 || status[target] == 5) continue;
            if (source != -1 && status[source] == 5) continue; // ISO
            
            if (runif(gen) < scale_func(t)) {
                if (runif(gen) >= get_efficacy(target, t)) {
                    if (!received_vaccine[target]) {
                        exposed_before_vaccination[target] = true;
                    }
                    status[target] = 1;
            exposure_time[target] = t;
                    total_infected++;
                    if (max_cases > 0 && total_infected >= max_cases) continue;
                    add_event(t + rexp_sigma(gen), 2, target);
                    continue;
                }
            }
            
            double eff_tau = tau_max;
            if (status[target] == 4 && risk_compensation_multiplier > 1.0) {
                eff_tau *= risk_compensation_multiplier;
            }
            std::exponential_distribution<> rexp_tau(eff_tau);
            double next_t = t + rexp_tau(gen);
            if (next_t < ev.rec_time) {
                add_event(next_t, 1, target, source, ev.rec_time);
            }
        }
        else if (ev.type == 2) { // ONSET
            if (status[target] == 1 || status[target] == 0) {
                if (status[target] == 1 && received_vaccine[target]) {
                    bool eligible = true;
                    if (!allow_pep && exposed_before_vaccination[target]) {
                        eligible = false;
                    }
                    if (eligible && runif(gen) < get_efficacy(target, t)) {
                        status[target] = 3;
                        total_infected--;
                        aborted_due_to_pep[target] = true;
                        recovery_or_death_time[target] = t;
                        continue;
                    }
                }
                
                if (status[target] == 0) total_infected++;
                status[target] = 2;
                onset_time[target] = t;
                double tb = get_therapeutic_efficacy(target, t);
                double cfr = base_CFR - tb * (base_CFR - vax_CFR);
                if (runif(gen) < cfr) {
                    total_deaths++;
                    died[target] = true;
                    recovery_or_death_time[target] = t;
                    if (return_time_series && t >= 0 && t <= max_sim_time) {
                        daily_deaths[int(t)]++;
                    }
                }

                double rec_t = t + rexp_gamma(gen);
                add_event(rec_t, 5, target); // 5: RECOVERY
                
                double current_rr = reporting_rate_scalar;
                if (!reporting_rate.empty()) {
                    int idx = std::min((int)t, (int)reporting_rate.size() - 1);
                    current_rr = reporting_rate[idx];
                }
                
                if (runif(gen) < current_rr) {
                    double det_time = t + (monitored[target] ? 1.0 : detection_delay);
                    if (det_time < rec_t) {
                        add_event(det_time, 3, target); // 3: DETECTION
                    }
                }
                
                for (int neighbor : adj[target]) {
                    if (status[neighbor] == 0 || status[neighbor] == 4) { // S or V
                        double eff_tau = tau_max;
                        if (status[neighbor] == 4 && risk_compensation_multiplier > 1.0) {
                            eff_tau *= risk_compensation_multiplier;
                        }
                        std::exponential_distribution<> rexp_tau(eff_tau);
                        double inf_time = t + rexp_tau(gen);
                        if (inf_time < rec_t) {
                            add_event(inf_time, 1, neighbor, target, rec_t);
                        }
                    }
                }
            }
        }
        else if (ev.type == 5) { // RECOVERY
            if (status[target] == 2 || status[target] == 5) {
                status[target] = 3;
                if (!died[target]) recovery_or_death_time[target] = t;
            }
        }
        else if (ev.type == 3) { // DETECTION
            if (t > max_sim_time) continue;
            
            if (!first_detection_occurred && community_vax_trigger == 2) {
                first_detection_occurred = true;
                schedule_community_vax(t + community_vax_delay);
            }
            
            if (t < vax_start_time) continue;
            if (status[target] == 2) status[target] = 5; // ISO

            // BFS for ring vaccination
            std::vector<int> distances(N, -1);
            std::queue<int> q;
            q.push(target);
            distances[target] = 0;
            
            std::vector<int> candidates;

            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                int d = distances[curr];
                if (d > 0) {
                    if (status[curr] == 0 || status[curr] == 1 || status[curr] == 4) { // S, E, or vaccinated susceptible
                        if (runif(gen) < get_trace_probability(d, t)) {
                            monitored[curr] = true;
                            if (scheduled_vaccine_nodes.find(curr) == scheduled_vaccine_nodes.end() && !received_vaccine[curr]) {
                                candidates.push_back(curr);
                            }
                        }
                    }
                }
                
                if (d < ring_radius) {
                    for (int neighbor : adj[curr]) {
                        if (distances[neighbor] == -1) {
                            distances[neighbor] = d + 1;
                            q.push(neighbor);
                        }
                    }
                }
            }

            if (compete_queue) {
                std::shuffle(candidates.begin(), candidates.end(), gen);
            }

            for (int curr : candidates) {
                int d = distances[curr];
                if (runif(gen) >= get_vaccine_acceptance(d)) continue;
                if (max_vaccines >= 0 && scheduled_vaccine_nodes.size() + total_vaccines >= max_vaccines) break;
                double trace_start_t = std::max(t, next_available_trace_time);
                double vax_t = trace_start_t + get_node_delay(d);
                next_available_trace_time = trace_start_t + (1.0 / max_daily_traces);
                
                scheduled_vaccine_nodes.insert(curr);
                add_event(vax_t, 4, curr); // 4: VACCINATION
            }
        }
        else if (ev.type == 4) { // VACCINATION
            if (t > max_sim_time) continue;
            scheduled_vaccine_nodes.erase(target);
state_at_vaccination[target] = status[target];
            if (status[target] == 0 || status[target] == 1 || status[target] >= 2) {
                if (!received_vaccine[target]) total_vaccines++;
                received_vaccine[target] = true;
                vaccination_time[target] = t;
            }
            if (status[target] == 0) status[target] = 4;
        }
        else if (ev.type == 6) { // COMMUNITY_VAX
            if (t > max_sim_time) continue;
            int num_to_vax = std::round(N * community_vax_coverage);
            int stop_index = std::min(num_to_vax, community_vax_next_index + (int)community_vax_daily_quota);
            for (int i = community_vax_next_index; i < stop_index; ++i) {
                int curr = community_vax_order[i];
                if (status[curr] >= 0) {
                    if (!received_vaccine[curr]) {
                        scheduled_vaccine_nodes.erase(curr);
                        total_vaccines++;
                        received_vaccine[curr] = true;
                        vaccination_time[curr] = t;
                        state_at_vaccination[curr] = status[curr];
                        if (status[curr] == 0) status[curr] = 4;
                    }
                }
            }
            community_vax_next_index = stop_index;
            if (community_vax_next_index < num_to_vax && community_vax_rollout_days > 0.0) {
                add_event(t + 1.0, 6, -1);
            }
        }
    }

    if (return_time_series) {
        return py::cast(daily_deaths);
    }
    
    py::dict res;
    res["exposure_time"] = exposure_time;
    res["onset_time"] = onset_time;
    res["recovery_or_death_time"] = recovery_or_death_time;
    res["died"] = died;
    res["aborted_due_to_pep"] = aborted_due_to_pep;
    res["state_at_vaccination"] = state_at_vaccination;
    res["vaccination_time"] = vaccination_time;
    res["total_infected"] = total_infected;
    res["total_deaths"] = total_deaths;
    res["total_vaccines"] = total_vaccines;
    
    return res;
}


PYBIND11_MODULE(ebola_stochastic_ring_old_cpp, m) {
    m.def("simulate_ring_vaccination_cpp", &simulate_ring_vaccination_cpp, pybind11::arg("N"), pybind11::arg("adj"), pybind11::arg("rt_array"), pybind11::arg("baseline_tau"), pybind11::arg("incubation_period"), pybind11::arg("infectious_period"), pybind11::arg("ring_radius"), pybind11::arg("efficacy"), pybind11::arg("immune_delay"), pybind11::arg("uptake"), pybind11::arg("reporting_rate"), pybind11::arg("reporting_rate_scalar"), pybind11::arg("detection_delay"), pybind11::arg("tracing_delay"), pybind11::arg("max_cases"), pybind11::arg("max_daily_traces"), pybind11::arg("max_vaccines"), pybind11::arg("base_CFR"), pybind11::arg("vax_CFR"), pybind11::arg("initial_infected"), pybind11::arg("initial_exposed"), pybind11::arg("max_sim_time"), pybind11::arg("vax_start_time"), pybind11::arg("return_time_series"), pybind11::arg("risk_compensation_multiplier"), pybind11::arg("trust_uptake_dependency"), pybind11::arg("tracing_coverage"), pybind11::arg("vaccine_acceptability"), pybind11::arg("sigmoidal_k"), pybind11::arg("sigmoidal_d0"), pybind11::arg("uptake_r2_drop"), pybind11::arg("tracing_delay_r2_add"), pybind11::arg("compete_queue"), pybind11::arg("allow_pep"), pybind11::arg("community_vax_coverage"), pybind11::arg("community_vax_trigger"), pybind11::arg("community_vax_delay"), pybind11::arg("community_vax_rollout_days"), pybind11::arg("seed"), pybind11::arg("infection_efficacy_multiplier") = 1.0, pybind11::arg("immediate_mortality_protection") = false, "C++ backend for Ebola simulation");
    m.def("simulate_mechanism_cpp", &simulate_mechanism_cpp, pybind11::arg("N"), pybind11::arg("adj"), pybind11::arg("rt_array"), pybind11::arg("baseline_tau"), pybind11::arg("incubation_period"), pybind11::arg("infectious_period"), pybind11::arg("ring_radius"), pybind11::arg("efficacy"), pybind11::arg("immune_delay"), pybind11::arg("uptake"), pybind11::arg("reporting_rate"), pybind11::arg("reporting_rate_scalar"), pybind11::arg("detection_delay"), pybind11::arg("tracing_delay"), pybind11::arg("max_cases"), pybind11::arg("max_daily_traces"), pybind11::arg("max_vaccines"), pybind11::arg("base_CFR"), pybind11::arg("vax_CFR"), pybind11::arg("initial_infected"), pybind11::arg("initial_exposed"), pybind11::arg("max_sim_time"), pybind11::arg("vax_start_time"), pybind11::arg("return_time_series"), pybind11::arg("risk_compensation_multiplier"), pybind11::arg("trust_uptake_dependency"), pybind11::arg("tracing_coverage"), pybind11::arg("vaccine_acceptability"), pybind11::arg("sigmoidal_k"), pybind11::arg("sigmoidal_d0"), pybind11::arg("uptake_r2_drop"), pybind11::arg("tracing_delay_r2_add"), pybind11::arg("compete_queue"), pybind11::arg("allow_pep"), pybind11::arg("community_vax_coverage"), pybind11::arg("community_vax_trigger"), pybind11::arg("community_vax_delay"), pybind11::arg("community_vax_rollout_days"), pybind11::arg("seed"), pybind11::arg("infection_efficacy_multiplier") = 1.0, pybind11::arg("immediate_mortality_protection") = false, "C++ mechanism backend");
}
