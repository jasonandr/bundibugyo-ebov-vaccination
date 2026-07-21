#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

struct Event {
    double t;
    long long id;
    int type; // 0: EXPOSURE, 2: ONSET, 3: DETECTION, 4: VACCINATION, 5: RECOVERY, 6: COMMUNITY_VAX
    int target;
    int source;
    double rec_time;

    bool operator<(const Event& other) const {
        if (t != other.t) return t > other.t; // Min-heap
        return id > other.id;
    }
};

class EbolaEngine {
private:
    int N;
    std::vector<int> adj_offsets;
    std::vector<int> adj_edges;

    // State buffers
    std::vector<int> status; // 0: S, 1: E, 2: I, 3: R, 4: V, 5: ISO
    std::vector<double> exposure_time;
    std::vector<double> onset_time;
    std::vector<double> recovery_or_death_time;
    std::vector<bool> died;
    std::vector<bool> aborted_due_to_pep;
    std::vector<int> state_at_vaccination;

    std::vector<bool> received_vaccine;
    std::vector<bool> monitored;
    std::vector<double> vaccination_time;
    std::vector<bool> exposed_before_vaccination;
    std::unordered_set<int> scheduled_vaccine_nodes;
    
    // Engine tracker for quick resets
    std::vector<int> touched_nodes;
    std::vector<bool> is_touched;

    // True Rt Tracking
    std::vector<double> true_rt_numerator;
    std::vector<double> true_rt_denominator;
    std::vector<double> daily_incidence;

    // For BFS
    std::vector<int> visit_id;
    int current_visit_id;
    std::vector<int> distances;

    // For community vax
    std::vector<int> community_vax_order;

    // Random streams
    std::mt19937 network_gen;
    std::mt19937 duration_gen;
    std::mt19937 transmission_gen;
    std::mt19937 detection_gen;
    std::mt19937 tracing_gen;
    std::mt19937 vaccine_gen;
    std::mt19937 mortality_gen;

    inline void mark_touched(int node) {
        if (!is_touched[node]) {
            is_touched[node] = true;
            touched_nodes.push_back(node);
        }
    }

public:
    EbolaEngine(int N, py::array_t<int> offsets_in, py::array_t<int> edges_in) : N(N) {
        auto buf_off = offsets_in.request();
        auto buf_edg = edges_in.request();
        adj_offsets.assign((int*)buf_off.ptr, ((int*)buf_off.ptr) + buf_off.size);
        adj_edges.assign((int*)buf_edg.ptr, ((int*)buf_edg.ptr) + buf_edg.size);

        status.assign(N, 0);
        exposure_time.assign(N, -1.0);
        onset_time.assign(N, -1.0);
        recovery_or_death_time.assign(N, -1.0);
        died.assign(N, false);
        aborted_due_to_pep.assign(N, false);
        state_at_vaccination.assign(N, -1);
        
        received_vaccine.assign(N, false);
        monitored.assign(N, false);
        vaccination_time.assign(N, -1.0);
        exposed_before_vaccination.assign(N, false);
        
        is_touched.assign(N, false);
        
        visit_id.assign(N, 0);
        distances.assign(N, -1);
        current_visit_id = 0;
    }

    void reset_state() {
        for (int node : touched_nodes) {
            status[node] = 0;
            exposure_time[node] = -1.0;
            onset_time[node] = -1.0;
            recovery_or_death_time[node] = -1.0;
            died[node] = false;
            aborted_due_to_pep[node] = false;
            state_at_vaccination[node] = -1;
            received_vaccine[node] = false;
            monitored[node] = false;
            vaccination_time[node] = -1.0;
            exposed_before_vaccination[node] = false;
            is_touched[node] = false;
        }
        touched_nodes.clear();
        scheduled_vaccine_nodes.clear();
    }

    py::object run_simulation(
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
        bool immediate_mortality_protection,
        bool return_mechanism,
        double incubation_shape,
        double infectious_shape,
        bool use_cohort
    ) {
        reset_state();

        int base_seed = (seed < 0) ? std::random_device{}() : seed;
        network_gen.seed(base_seed ^ 0x1234567);
        duration_gen.seed(base_seed ^ 0x7654321);
        transmission_gen.seed(base_seed ^ 0xABCDEF);
        detection_gen.seed(base_seed ^ 0xFEDCBA);
        tracing_gen.seed(base_seed ^ 0x112233);
        vaccine_gen.seed(base_seed ^ 0x445566);
        mortality_gen.seed(base_seed ^ 0x778899);

        std::uniform_real_distribution<> runif_trans(0.0, 1.0);
        std::uniform_real_distribution<> runif_vax(0.0, 1.0);
        std::uniform_real_distribution<> runif_det(0.0, 1.0);
        std::uniform_real_distribution<> runif_trace(0.0, 1.0);
        std::uniform_real_distribution<> runif_mort(0.0, 1.0);

        std::gamma_distribution<> rgamma_gamma(infectious_shape, infectious_period / infectious_shape);
        std::gamma_distribution<> rgamma_sigma(incubation_shape, incubation_period / incubation_shape);

        double R_max = rt_array.empty() ? 1.66 : 0.0;
        if (!rt_array.empty()) {
            for (double v : rt_array) if (v > R_max) R_max = v;
        }
        R_max = std::max(R_max, 0.01);

        auto get_rt = [&](double t) {
            if (!rt_array.empty()) {
                int idx = static_cast<int>(t);
                if (idx <= 0) return rt_array[0];
                if (idx >= (int)rt_array.size()) return rt_array.back();
                return rt_array[idx];
            }
            return 1.66;
        };

        auto scale_func = [&](double t) { return get_rt(t) / R_max; };

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
        std::iota(nodes.begin(), nodes.end(), 0);
        std::shuffle(nodes.begin(), nodes.end(), network_gen);
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
        
        true_rt_numerator.assign(max_sim_time + 2, 0.0);
        true_rt_denominator.assign(max_sim_time + 2, 0.0);
        daily_incidence.assign(max_sim_time + 2, 0.0);

        int community_vax_next_index = 0;
        double community_vax_daily_quota = 0.0;

        auto schedule_community_vax = [&](double start_t) {
            if (community_vax_coverage <= 0.0) return;
            int num_to_vax = std::round(N * community_vax_coverage);
            if (num_to_vax <= 0) return;
            community_vax_order.resize(N);
            std::iota(community_vax_order.begin(), community_vax_order.end(), 0);
            std::shuffle(community_vax_order.begin(), community_vax_order.end(), network_gen);
            community_vax_next_index = 0;
            if (community_vax_rollout_days > 0.0) {
                community_vax_daily_quota = std::max(1.0, std::ceil(num_to_vax / community_vax_rollout_days));
            } else {
                community_vax_daily_quota = num_to_vax;
            }
            add_event(start_t, 6, -1);
        };

        bool first_detection_occurred = false;
        if (community_vax_trigger == 1) {
            schedule_community_vax(community_vax_delay);
        }

        for (int n : initial_nodes) {
            mark_touched(n);
            add_event(-detection_delay, 2, n);
        }
        for (int n : initial_exposed_nodes) {
            mark_touched(n);
            status[n] = 1; // E
            exposed_before_vaccination[n] = true;
            total_infected++;
            add_event(rgamma_sigma(duration_gen), 2, n);
        }

        double last_event_t = 0.0;
        while (!pq.empty()) {
            Event ev = pq.top();
            pq.pop();
            double t = ev.t;
            if (t <= max_sim_time) {
                last_event_t = std::max(last_event_t, t);
            }
            int target = ev.target;
            int source = ev.source;

            if (ev.type == 0) { // EXPOSURE
                if (t > max_sim_time) continue;
                if (source != -1 && status[source] == 5) continue; // Blocked by isolation!
                if (status[target] == 1 || status[target] == 2 || status[target] == 3 || status[target] == 5) continue;
                
                // Track true Rt for legacy non-cohort engine
                if (!use_cohort && source != -1) {
                    int int_t = static_cast<int>(t);
                    if (int_t >= 0 && int_t <= max_sim_time) {
                        true_rt_numerator[int_t] += 1.0;
                    }
                }
                
                if (runif_vax(vaccine_gen) < get_efficacy(target, t)) continue;
                
                mark_touched(target);
                if (!received_vaccine[target]) {
                    exposed_before_vaccination[target] = true;
                }
                status[target] = 1;
                if (return_mechanism) exposure_time[target] = t;
                total_infected++;
                if (max_cases > 0 && total_infected >= max_cases) continue;

                add_event(t + rgamma_sigma(duration_gen), 2, target);
            } 
            else if (ev.type == 2) { // ONSET
                mark_touched(target);
                if (status[target] == 1 || status[target] == 0) {
                    if (status[target] == 1 && received_vaccine[target]) {
                        bool eligible = true;
                        if (!allow_pep && exposed_before_vaccination[target]) {
                            eligible = false;
                        }
                        if (eligible && runif_vax(vaccine_gen) < get_efficacy(target, t)) {
                            status[target] = 3; // R
                            total_infected--;
                            if (return_mechanism) {
                                aborted_due_to_pep[target] = true;
                                recovery_or_death_time[target] = t;
                            }
                            continue;
                        }
                    }
                    
                    if (status[target] == 0) total_infected++;
                    status[target] = 2; // I
                    if (return_mechanism) onset_time[target] = t;
                    double tb = get_therapeutic_efficacy(target, t);
                    double cfr = base_CFR - tb * (base_CFR - vax_CFR);
                    if (runif_mort(mortality_gen) < cfr) {
                        total_deaths++;
                        if (return_mechanism) {
                            died[target] = true;
                            recovery_or_death_time[target] = t;
                        }
                        if (return_time_series && t >= 0 && t <= max_sim_time) {
                            daily_deaths[int(t)]++;
                        }
                    }

                    double rec_t = t + rgamma_gamma(duration_gen);
                    add_event(rec_t, 5, target); // RECOVERY
                    
                    double current_rr = reporting_rate_scalar;
                    if (!reporting_rate.empty()) {
                        int idx = std::min((int)t, (int)reporting_rate.size() - 1);
                        current_rr = reporting_rate[idx];
                    }
                    
                    if (runif_det(detection_gen) < current_rr) {
                        double det_time = t + (monitored[target] ? 1.0 : detection_delay);
                        if (det_time < rec_t) {
                            add_event(det_time, 3, target); // DETECTION
                        }
                    }
                    
                    int int_t = static_cast<int>(t);
                    if (int_t >= 0 && int_t <= max_sim_time) {
                        daily_incidence[int_t] += 1.0;
                        if (!use_cohort) {
                            true_rt_denominator[int_t] += 1.0; // Track legacy denominator
                        }
                    }

                    // --- EFFICIENT INFECTION GENERATION ---
                    if (use_cohort) {
                        int start_edge = adj_offsets[target];
                        int end_edge = adj_offsets[target + 1];
                        std::vector<int> susceptible_neighbors;
                        susceptible_neighbors.reserve(end_edge - start_edge);
                        for (int e = start_edge; e < end_edge; ++e) {
                            int nbr = adj_edges[e];
                            if (status[nbr] == 0 || status[nbr] == 4) {
                                susceptible_neighbors.push_back(nbr);
                            }
                        }
                        if (!susceptible_neighbors.empty()) {
                            double target_rt = get_rt(t);
                            double p = std::min(1.0, target_rt / static_cast<double>(susceptible_neighbors.size()));
                            std::binomial_distribution<int> rbinom(susceptible_neighbors.size(), p);
                            int num_infections = rbinom(transmission_gen);
                            if (int_t >= 0 && int_t <= max_sim_time) {
                                true_rt_numerator[int_t] += static_cast<double>(num_infections);
                                true_rt_denominator[int_t] += 1.0;
                            }
                            if (num_infections > 0) {
                                std::shuffle(susceptible_neighbors.begin(), susceptible_neighbors.end(), transmission_gen);
                                std::uniform_real_distribution<> runif_inf(t, rec_t);
                                for (int i = 0; i < num_infections; ++i) {
                                    int sec_target = susceptible_neighbors[i];
                                    double inf_time = runif_inf(transmission_gen);
                                    add_event(inf_time, 0, sec_target, target, rec_t);
                                }
                            }
                        }
                    } else {
                        double base_prob = scale_func(t);
                        double eff_tau = tau_max;
                        if (status[target] == 4 && risk_compensation_multiplier > 1.0) {
                            eff_tau *= risk_compensation_multiplier;
                        }
                        
                        int start_edge = adj_offsets[target];
                        int end_edge = adj_offsets[target + 1];
                        int degree = end_edge - start_edge;
                        
                        if (degree > 0) {
                            double lambda = degree * eff_tau * base_prob * (rec_t - t);
                            std::poisson_distribution<int> rpois(lambda);
                            int num_contacts = rpois(transmission_gen);
                            
                            if (num_contacts > 0) {
                                std::uniform_real_distribution<> runif_inf(t, rec_t);
                                std::uniform_int_distribution<> runif_neighbor(start_edge, end_edge - 1);
                                
                                for (int c = 0; c < num_contacts; ++c) {
                                    int neighbor = adj_edges[runif_neighbor(transmission_gen)];
                                    double inf_time = runif_inf(transmission_gen);
                                    add_event(inf_time, 0, neighbor, target, rec_t);
                                }
                            }
                        }
                    }
                }
            }
            else if (ev.type == 5) { // RECOVERY
                mark_touched(target);
                if (status[target] == 2 || status[target] == 5) {
                    status[target] = 3;
                    if (return_mechanism && !died[target]) recovery_or_death_time[target] = t;
                }
            }
            else if (ev.type == 3) { // DETECTION
                if (t > max_sim_time) continue;
                
                if (!first_detection_occurred && community_vax_trigger == 2) {
                    first_detection_occurred = true;
                    schedule_community_vax(t + community_vax_delay);
                }
                
                if (t < vax_start_time) continue;
                mark_touched(target);
                if (status[target] == 2) status[target] = 5; // ISO

                // BFS for ring vaccination
                current_visit_id++;
                std::queue<int> q;
                q.push(target);
                visit_id[target] = current_visit_id;
                distances[target] = 0;
                
                std::vector<int> candidates;

                while (!q.empty()) {
                    int curr = q.front();
                    q.pop();
                    
                    int d = distances[curr];
                    if (d > 0) {
                        if (status[curr] == 0 || status[curr] == 1 || status[curr] == 4) {
                            if (runif_trace(tracing_gen) < get_trace_probability(d, t)) {
                                mark_touched(curr);
                                monitored[curr] = true;
                                if (scheduled_vaccine_nodes.find(curr) == scheduled_vaccine_nodes.end() && !received_vaccine[curr]) {
                                    candidates.push_back(curr);
                                }
                            }
                        }
                    }
                    
                    if (d < ring_radius) {
                        int start_edge = adj_offsets[curr];
                        int end_edge = adj_offsets[curr + 1];
                        for (int idx = start_edge; idx < end_edge; ++idx) {
                            int neighbor = adj_edges[idx];
                            if (visit_id[neighbor] != current_visit_id) {
                                visit_id[neighbor] = current_visit_id;
                                distances[neighbor] = d + 1;
                                q.push(neighbor);
                            }
                        }
                    }
                }

                if (compete_queue) {
                    std::shuffle(candidates.begin(), candidates.end(), tracing_gen);
                }

                for (int curr : candidates) {
                    int d = distances[curr];
                    if (runif_vax(vaccine_gen) >= get_vaccine_acceptance(d)) continue;
                    if (max_vaccines >= 0 && scheduled_vaccine_nodes.size() + total_vaccines >= max_vaccines) break;
                    double trace_start_t = std::max(t, next_available_trace_time);
                    double vax_t = trace_start_t + get_node_delay(d);
                    next_available_trace_time = trace_start_t + (1.0 / max_daily_traces);
                    
                    scheduled_vaccine_nodes.insert(curr);
                    add_event(vax_t, 4, curr); // VACCINATION
                }
            }
            else if (ev.type == 4) { // VACCINATION
                if (t > max_sim_time) continue;
                scheduled_vaccine_nodes.erase(target);
                mark_touched(target);
                if (return_mechanism) state_at_vaccination[target] = status[target];
                if (status[target] == 0 || status[target] == 1 || (return_mechanism && status[target] >= 2)) {
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
                            mark_touched(curr);
                            scheduled_vaccine_nodes.erase(curr);
                            total_vaccines++;
                            received_vaccine[curr] = true;
                            vaccination_time[curr] = t;
                            if (return_mechanism) state_at_vaccination[curr] = status[curr];
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
            py::dict res;
            res["daily_deaths"] = daily_deaths;
            res["daily_incidence"] = daily_incidence;
            res["true_rt_numerator"] = true_rt_numerator;
            res["true_rt_denominator"] = true_rt_denominator;
            return res;
        }
        
        if (return_mechanism) {
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

        return py::make_tuple((double)total_infected / N, (double)total_deaths / N, total_vaccines, last_event_t);
    }
};

PYBIND11_MODULE(ebola_stochastic_ring_cpp, m) {
    py::class_<EbolaEngine>(m, "EbolaEngine")
        .def(py::init<int, py::array_t<int>, py::array_t<int>>())
        .def("run_simulation", &EbolaEngine::run_simulation,
             py::arg("rt_array"), py::arg("baseline_tau"), py::arg("incubation_period"),
             py::arg("infectious_period"), py::arg("ring_radius"), py::arg("efficacy"),
             py::arg("immune_delay"), py::arg("uptake"), py::arg("reporting_rate"),
             py::arg("reporting_rate_scalar"), py::arg("detection_delay"), py::arg("tracing_delay"),
             py::arg("max_cases"), py::arg("max_daily_traces"), py::arg("max_vaccines"),
             py::arg("base_CFR"), py::arg("vax_CFR"), py::arg("initial_infected"),
             py::arg("initial_exposed"), py::arg("max_sim_time"), py::arg("vax_start_time"),
             py::arg("return_time_series"), py::arg("risk_compensation_multiplier"),
             py::arg("trust_uptake_dependency"), py::arg("tracing_coverage"),
             py::arg("vaccine_acceptability"), py::arg("sigmoidal_k"), py::arg("sigmoidal_d0"),
             py::arg("uptake_r2_drop"), py::arg("tracing_delay_r2_add"), py::arg("compete_queue"),
             py::arg("allow_pep"), py::arg("community_vax_coverage"), py::arg("community_vax_trigger"),
             py::arg("community_vax_delay"), py::arg("community_vax_rollout_days"), py::arg("seed"),
             py::arg("infection_efficacy_multiplier") = 1.0, py::arg("immediate_mortality_protection") = false,
             py::arg("return_mechanism") = false,
             py::arg("incubation_shape") = 1.0, py::arg("infectious_shape") = 1.0,
             py::arg("use_cohort") = true,
             "Run a fast simulation replicate using persistent arrays."
        );
}
