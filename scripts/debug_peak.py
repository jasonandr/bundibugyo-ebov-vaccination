import numpy as np
import json
from queue import PriorityQueue

with open('../data_and_results/fitted_parameters.json', 'r') as f:
    params = json.load(f)
rt_array = np.array(params.get('Rt_array', []))
max_sim_time = 100
if len(rt_array) < max_sim_time:
    rt_array = list(rt_array) + [rt_array[-1]] * (max_sim_time - len(rt_array))

def get_rt(t):
    idx = int(t)
    if idx < 0: return rt_array[0]
    if idx >= max_sim_time: return rt_array[-1]
    return rt_array[idx]

inc_reps = []
for rep in range(100):
    pq = PriorityQueue()
    daily_incidence = np.zeros(max_sim_time + 1)
    num = np.zeros(max_sim_time + 1)
    den = np.zeros(max_sim_time + 1)
    
    for i in range(5):
        pq.put((-4.0, 'ONSET'))
    for i in range(5):
        pq.put((np.random.exponential(8.5), 'ONSET'))
        
    while not pq.empty():
        t, ev = pq.get()
        if t > max_sim_time: continue
        
        if ev == 'ONSET':
            if t >= 0:
                daily_incidence[int(t)] += 1
            
            rec_t = t + np.random.exponential(6.0)
            target_rt = get_rt(t)
            
            infections = np.random.poisson(target_rt)
            
            if t >= 0:
                num[int(t)] += infections
                den[int(t)] += 1
                
            for _ in range(infections):
                inf_time = t + np.random.uniform(0, rec_t - t)
                onset_time = inf_time + np.random.exponential(8.5)
                pq.put((onset_time, 'ONSET'))
                
    inc_reps.append(daily_incidence)

mean_inc = np.mean(inc_reps, axis=0)
print("Day | Queue Branching Inc")
for day in range(0, 80, 5):
    print(f"{day:3d} | {mean_inc[day]:8.2f}")
