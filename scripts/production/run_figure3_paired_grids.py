"""Paired current-engine Figure 3 grids; each cell shares seeds with its comparator."""
import argparse,csv,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from ebola_stochastic_ring import simulate_ring_vaccination
from network_cache import load_cached_network
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parent.parent; G=P=RT=None; N=100000
def init(params,rt,cache):
 global G,P,RT; G,P,RT=load_cached_network(cache),params,rt
def ramp(x): return np.linspace(.3,x,15).tolist()+[x]*76
def sim(s,seed):
 rr=ramp(s['det']) if s.get('ramp') else [.3]*91; tr=ramp(s['trace']) if s.get('ramp') else [.3]*91
 return simulate_ring_vaccination(G,rt_array=RT,incubation_period=8.5,infectious_period=6,ring_radius=2,efficacy=s.get('ve',0),uptake=.8,reporting_rate=rr,tracing_coverage=tr,vaccine_acceptability=1,detection_delay=4,tracing_delay=2,max_daily_traces=100,max_vaccines=s.get('max_vaccines',0),base_CFR=P['base_CFR'],initial_infected=15,initial_exposed=15,max_sim_time=90,seed=seed,engine='cpp',community_vax_coverage=s.get('coverage',0),community_vax_trigger=1 if s.get('coverage',0) else 0,community_vax_delay=0 if s.get('coverage',0) else -1,community_vax_rollout_days=14 if s.get('coverage',0) else 0,risk_compensation_multiplier=s.get('risk',1),sigmoidal_k=.5,sigmoidal_d0=10,allow_pep=True,incubation_shape=2,infectious_shape=2)
def run(t):
 name,spec,reps,seed0=t; base={'det':.3,'trace':.3,'max_vaccines':0}
 out=[]
 for r in range(reps):
  seed=seed0+r; b=sim(base,seed)[1]*N; x=sim(spec,seed)[1]*N
  out.append({**spec,'analysis':name,'replicate':r,'seed':seed,'base_deaths':b,'scenario_deaths':x,'mortality_reduction_pct':100*(b-x)/b})
 return out
def main():
 q=argparse.ArgumentParser();q.add_argument('--output-dir',type=Path,required=True);q.add_argument('--network-cache',type=Path,required=True);q.add_argument('--workers',type=int,default=8);q.add_argument('--replicates',type=int,default=100);a=q.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 p=json.load(open(REPO/'data_and_results/fitted_parameters.json'));rt=p['Rt_array']+[p['Rt_array'][-1]]*25;ts=[]
 for d in np.linspace(0,1,11):
  for tr in np.linspace(0,1,11):ts.append(('fig3_operations',{'det':float(d),'trace':float(tr),'ramp':True,'max_vaccines':0},a.replicates,2026200000+len(ts)*1000))
 for c in np.linspace(0,.8,9):
  for v in np.linspace(0,.9,7):ts.append(('fig3_community_ve',{'det':.3,'trace':.3,'coverage':float(c),'ve':float(v),'max_vaccines':0},a.replicates,2026300000+len(ts)*1000))
 for rsk in np.linspace(1,2.5,7):
  for v in np.linspace(0,.9,7):ts.append(('fig3_risk_compensation',{'det':.3,'trace':.3,'coverage':.4,'ve':float(v),'risk':float(rsk),'max_vaccines':0},a.replicates,2026400000+len(ts)*1000))
 a.output_dir.mkdir(parents=True); rows=[]
 with mp.Pool(a.workers,initializer=init,initargs=(p,rt,str(a.network_cache))) as pool:
  for z in pool.imap_unordered(run,ts):rows+=z
 for name in set(r['analysis'] for r in rows):
  x=[r for r in rows if r['analysis']==name]
  with (a.output_dir/f'{name}_paired_raw.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in x for k in r}));w.writeheader();w.writerows(x)
if __name__=='__main__':main()
