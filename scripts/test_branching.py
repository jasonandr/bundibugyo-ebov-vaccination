import numpy as np
import json

with open('../data_and_results/fitted_parameters.json', 'r') as f:
    params = json.load(f)
rt_array = np.array(params.get('Rt_array', []))

mean_inc = np.zeros(100)
for rep in range(1000):
    onsets_by_day = [0] * 200
    onsets_by_day[0] = 5
    
    for t in range(100):
        # Number of people having onset at day t
        cases_today = onsets_by_day[t]
        if cases_today == 0: continue
        
        # Each case generates a Poisson(Rt) number of secondary cases
        rt = rt_array[t] if t < len(rt_array) else rt_array[-1]
        for _ in range(cases_today):
            # Draw infectious period
            inf_dur = np.random.exponential(6.0)
            # lambda is exactly Rt
            secondary = np.random.poisson(rt)
            
            for _ in range(secondary):
                # When do they get infected?
                inf_time = t + np.random.uniform(0, inf_dur)
                # When is their onset?
                onset_time = inf_time + np.random.exponential(8.5)
                
                day = int(onset_time)
                if day < 200:
                    onsets_by_day[day] += 1
                    
    mean_inc += np.array(onsets_by_day[:100])

mean_inc /= 1000.0
print("Day | Expected Branching Inc")
for day in range(0, 80, 5):
    print(f"{day:3d} | {mean_inc[day]:8.2f}")
