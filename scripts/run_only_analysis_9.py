import run_final_high_replicate_estimates as rf
_orig_scenario_definitions = rf.scenario_definitions
rf.scenario_definitions = lambda: [s for s in _orig_scenario_definitions() if s['scenario'] == 'analysis_9_contour_det_trace_r2']
if __name__ == '__main__':
    rf.main()
