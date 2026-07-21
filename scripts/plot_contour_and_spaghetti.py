import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import datetime
import os
import json

import ebola_stochastic_ring as sim
from paths import result_path

def init_worker():
    global _G
    _G = sim.generate_network(10000)
    global TAU_ARRAY
    with open(result_path("rt_calibrated_tau_array.json")) as f:
        TAU_ARRAY = json.load(f)["tau_array"]

# --- SPAGHETTI PLOT LOGIC ---
def plot_spaghetti_from_chunks():
    print("Loading spaghetti chunks from data_and_results/spaghetti_chunks...")
    all_nv_mod = []
    all_nv_opt = []
    all_r2_mod = []
    all_r2_opt = []
    all_c40 = []
    all_hyb = []
    
    for i in range(1, 101):
        path = f"data_and_results/spaghetti_chunks/chunk_{i}.npz"
        if os.path.exists(path):
            data = np.load(path)
            all_nv_mod.append(data['no_vax_mod'])
            all_nv_opt.append(data['no_vax_opt'])
            all_r2_mod.append(data['ring2_mod'])
            all_r2_opt.append(data['ring2_opt'])
            all_c40.append(data['comm_40'])
            all_hyb.append(data['hybrid'])
            
    if not all_nv_mod:
        print("No spaghetti data chunks found!")
        return
        
    def pad_to_max(arr_list):
        all_1d = []
        for arr in arr_list:
            if arr.ndim == 1:
                if arr.dtype == object:
                    all_1d.extend(list(arr))
                else:
                    all_1d.append(arr)
            elif arr.ndim == 2:
                all_1d.extend(list(arr))
        
        if not all_1d:
            return np.array([])
            
        max_len = max(len(a) for a in all_1d)
        padded = []
        for a in all_1d:
            if len(a) < max_len:
                padded.append(np.pad(a, (0, max_len - len(a)), 'constant'))
            else:
                padded.append(a)
        return np.vstack(padded)

    nv_mod = pad_to_max(all_nv_mod)
    nv_opt = pad_to_max(all_nv_opt)
    r2_mod = pad_to_max(all_r2_mod)
    r2_opt = pad_to_max(all_r2_opt)
    c40 = pad_to_max(all_c40)
    hyb = pad_to_max(all_hyb)
    
    n_sims = len(nv_mod)
    max_days = 90
    days = np.arange(max_days + 1)
    
    def calc_averted_percent(no_vax, vax):
        res = []
        for i in range(len(no_vax)):
            base = no_vax[i][:max_days+1] if len(no_vax[i]) > max_days else np.pad(no_vax[i], (0, max_days + 1 - len(no_vax[i])), 'edge')
            v = vax[i][:max_days+1] if len(vax[i]) > max_days else np.pad(vax[i], (0, max_days + 1 - len(vax[i])), 'edge')
            baseline_total = max(np.cumsum(base)[-1], 1)
            res.append((np.cumsum(base) - np.cumsum(v)) / baseline_total * 100.0)
        return np.array(res)

    def calc_averted_absolute(no_vax, vax):
        res = []
        for i in range(len(no_vax)):
            base = no_vax[i][:max_days+1] if len(no_vax[i]) > max_days else np.pad(no_vax[i], (0, max_days + 1 - len(no_vax[i])), 'edge')
            v = vax[i][:max_days+1] if len(vax[i]) > max_days else np.pad(vax[i], (0, max_days + 1 - len(vax[i])), 'edge')
            res.append(np.cumsum(base) - np.cumsum(v))
        return np.array(res)

    def calc_cumulative(no_vax, vax):
        res = []
        for i in range(len(vax)):
            v = vax[i][:max_days+1] if len(vax[i]) > max_days else np.pad(vax[i], (0, max_days + 1 - len(vax[i])), 'edge')
            res.append(np.cumsum(v))
        return np.array(res)

    def generate_plot(metric_func, ylabel, filename_suffix, ylim_bounds):
        averted_r2_mod = metric_func(nv_mod, r2_mod)
        averted_r2_opt = metric_func(nv_opt, r2_opt)
        averted_c40 = metric_func(nv_mod, c40)
        averted_hyb = metric_func(nv_mod, hyb)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
        axes = axes.flatten()
        
        scenarios_to_plot = [
            (averted_r2_mod, '#4f6d7a', "A"),
            (averted_c40, '#2a9d8f', "B"),
            (averted_hyb, '#e76f51', "C")
        ]
        
        for ax, (data, color, title) in zip(axes, scenarios_to_plot):
            final_averted = data[:, -1]
            q25, q75 = np.percentile(final_averted, 25), np.percentile(final_averted, 75)
            in_iqr = (final_averted >= q25) & (final_averted <= q75)
            
            plotted_count = 0
            for i in range(n_sims):
                if in_iqr[i]:
                    ax.plot(days, data[i], color=color, alpha=0.015, linewidth=1)
                    plotted_count += 1
                    if plotted_count >= 1000: break
            
            ax.plot(days, np.median(data, axis=0)[:max_days+1], color=color, linewidth=2.5, label="Median (IQR subset)")
            
            ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Days since outbreak start")
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_ylim(ylim_bounds)
        
        plt.tight_layout()
        path = f"figures/new_analyses/fig_spaghetti_{filename_suffix}.png"
        plt.savefig(path)
        print(f"SPAGHETTI_{filename_suffix}={path}")
        
    def calc_change_percent(no_vax, vax):
        res = []
        for i in range(len(no_vax)):
            base = no_vax[i][:max_days+1] if len(no_vax[i]) > max_days else np.pad(no_vax[i], (0, max_days + 1 - len(no_vax[i])), 'edge')
            v = vax[i][:max_days+1] if len(vax[i]) > max_days else np.pad(vax[i], (0, max_days + 1 - len(vax[i])), 'edge')
            baseline_total = max(np.cumsum(base)[-1], 1)
            res.append((np.cumsum(v) - np.cumsum(base)) / baseline_total * 100.0)
        return np.array(res)
        
    generate_plot(calc_change_percent, "Change in mortality (%)", "4panel", ylim_bounds=(-80, 20))

# --- CONTOUR PLOT LOGIC ---
def run_contour_point(args):
    cov, eff, seed = args
    global _G, TAU_ARRAY
    enh_reporting = np.linspace(0.3, 0.7, 15).tolist() + [0.7]*76
    enh_tracing = np.linspace(0.3, 0.8, 15).tolist() + [0.8]*76
    
    no_vax = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25,
        efficacy=0.0, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        max_sim_time=90, seed=seed, engine='cpp'
    )
    
    comm = sim.simulate_ring_vaccination(
        _G, initial_infected=5, rt_array=TAU_ARRAY, ring_radius=2, baseline_tau=0.25, max_vaccines=0,
        efficacy=eff, reporting_rate=enh_reporting, tracing_coverage=enh_tracing,
        community_vax_coverage=cov, community_vax_trigger=2, community_vax_delay=0.0,
        max_sim_time=90, seed=seed, engine='cpp'
    )
    return no_vax[1], comm[1]

def plot_contour():
    coverages = np.linspace(0.1, 0.8, 15) # smoother
    efficacies = np.linspace(0.3, 1.0, 15) # smoother
    n_reps = 1000
    
    args = []
    for cov in coverages:
        for eff in efficacies:
            for rep in range(n_reps):
                args.append((cov, eff, rep))
                
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"Running contour map with {len(args)} simulations...")
    with Pool(processes=workers, initializer=init_worker) as pool:
        results = pool.map(run_contour_point, args)
        
    idx = 0
    Z_averted = np.zeros((len(efficacies), len(coverages)))
    for i, cov in enumerate(coverages):
        for j, eff in enumerate(efficacies):
            chunk = results[idx:idx+n_reps]
            averted = []
            for no_vax, vax in chunk:
                if no_vax > 0:
                    averted.append((vax - no_vax) / no_vax * 100)
            Z_averted[j, i] = np.median(averted) if averted else 0
            idx += n_reps
            
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    X, Y = np.meshgrid(coverages * 100, efficacies * 100)
    
    from scipy.ndimage import gaussian_filter
    Z_averted_smooth = gaussian_filter(Z_averted, sigma=1.2)
    
    max_val = max(0, Z_averted_smooth.max())
    contour = ax.contourf(X, Y, Z_averted_smooth, levels=np.arange(-100, max_val + 5, 5), cmap='magma')
    plt.colorbar(contour, ax=ax, label="Median Change in Mortality (%)", ticks=np.arange(-100, 20, 10))
    
    # Overlaid contour lines at -25%, -50%, -75%
    line_contour = ax.contour(X, Y, Z_averted_smooth, levels=[-75, -50, -25], colors='black', linewidths=1.5)
    ax.clabel(line_contour, inline=True, fontsize=10, fmt='%1.0f%%')
    
    ax.set_title("Figure 5. Sensitivity of Community Mass Vaccination")
    ax.set_xlabel("Community Vaccination Coverage (%)")
    ax.set_ylabel("Vaccine Efficacy (%)")
    
    plt.tight_layout()
    path = f"figures/new_analyses/fig5_contour.png"
    plt.savefig(path)
    print(f"CONTOUR={path}")

if __name__ == "__main__":
    _G = sim.generate_network(10000)
    plot_spaghetti_from_chunks()
    plot_contour()

