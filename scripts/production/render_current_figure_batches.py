"""Summarise and render review-only figures from current-engine batch outputs."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import numpy as np


def read(path):
    with path.open() as f: return list(csv.DictReader(f))
def f(row, key): return float(row[key])
def grouped(rows, keys):
    out={}
    for r in rows:
        values=[]
        for k in keys:
            try: values.append(round(f(r,k), 8))
            except ValueError: values.append(r[k])
        out.setdefault(tuple(values),[]).append(r)
    return out
def median_deaths(rows):
    key="deaths" if "deaths" in rows[0] else "scenario_deaths"
    return np.median([f(r,key) for r in rows])
def reduction(rows, comparator): return 100*(comparator-median_deaths(rows))/comparator
def save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white")
    pdf = path.with_suffix(".pdf") if hasattr(path, "with_suffix") else None
    if pdf is not None: fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output_dir",type=Path,help="Directory containing the Figure 4–5 raw outputs.")
    p.add_argument("--risk-raw",type=Path,help="Current weighted Figure 3C raw output.")
    p.add_argument("--paired-dir",type=Path,help="Directory containing paired Figure 3A/B raw outputs.")
    p.add_argument("--fig4-onset-paired-raw",type=Path,help="Same-seed paired Figure 4C raw output.")
    p.add_argument("--render-dir",type=Path,help="Destination for review renders (defaults to output_dir).")
    p.add_argument("--include-figure5",action="store_true",help="Render the exploratory Figure 5 timing plot.")
    a=p.parse_args(); root=a.output_dir; render=a.render_dir or root
    if not root.exists(): raise FileNotFoundError(root)
    render.mkdir(parents=True,exist_ok=True)
    rc={"font.family":"DejaVu Sans","font.size":10}; plt.rcParams.update(rc)
    summary=[]
    # Figure 3
    op_path=root/"fig3_operations_raw.csv"
    if a.paired_dir: op_path=a.paired_dir/"fig3_operations_paired_raw.csv"
    op=read(op_path); op_keys=("detection","tracing") if "detection" in op[0] else ("det","trace"); go=grouped(op,op_keys); base=median_deaths(go[(.3,.3)])
    det=np.array(sorted({k[0] for k in go})); tr=np.array(sorted({k[1] for k in go}))
    za=np.array([[np.median([f(x,"mortality_reduction_pct") for x in go[(d,t)]]) if "mortality_reduction_pct" in op[0] else reduction(go[(d,t)],base) for d in det] for t in tr])
    comm_path=root/"fig3_community_ve_raw.csv"
    if a.paired_dir: comm_path=a.paired_dir/"fig3_community_ve_paired_raw.csv"
    comm=read(comm_path); gc=grouped(comm,("coverage","ve" if "ve" in comm[0] else "ve_i")); base_c=median_deaths(gc[(0.0,.45)])
    cov=np.array(sorted({k[0] for k in gc})); ve=np.array(sorted({k[1] for k in gc})); zb=np.array([[np.median([f(x,"mortality_reduction_pct") for x in gc[(c,v)]]) if "mortality_reduction_pct" in comm[0] else reduction(gc[(c,v)],base_c) for c in cov] for v in ve])
    risk=read(a.risk_raw or root/"fig3_risk_compensation_raw.csv"); gr=grouped(risk,("risk","ve" if "ve" in risk[0] else "ve_i")); zr=np.array([[np.median([f(x,"mortality_reduction_pct") for x in gr[(r,v)]]) if "mortality_reduction_pct" in risk[0] else reduction(gr[(r,v)],base_c) for r in sorted({k[0] for k in gr})] for v in ve]); risks=np.array(sorted({k[0] for k in gr}))
    for name, grid, xs, ys in [("fig3_operations",za,det,tr),("fig3_community_ve",zb,cov,ve),("fig3_risk_compensation",zr,risks,ve)]:
        for j,y in enumerate(ys):
            for i,x in enumerate(xs): summary.append({"analysis":name,"x":x,"y":y,"median_mortality_reduction_pct":grid[j,i]})
    benefit=LinearSegmentedColormap.from_list("benefit",["#A84F3D","#D88762","#F0D7B3","#F7F4EA","#B8C9B2","#60908D","#285A6E"])
    vaccine=LinearSegmentedColormap.from_list("vaccine",["#F5F0D8","#D6CF8A","#9BB56D","#4F927E","#246477"])
    diverging=LinearSegmentedColormap.from_list("div",["#8F3028","#C95D48","#E6A17B","#F4E7CD","#D7E2D2","#88AAA1","#326879"])
    risk_mask = risks <= 2.0
    risk_ve_mask = ve >= .15
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.6)); panels=[(za,det*100,tr*100,"Index-case detection (%)","Contact tracing coverage (%)","A",benefit,[-80,-60,-40,-25,-10,0,5,12,22,35,50,65,80,90]),(zb,cov*100,ve*100,"Community vaccination coverage (%)","Vaccine effectiveness (%)","B",vaccine,[0,2,5,9,14,20,28,38,50,62,74,84,92,96]),(zr[np.ix_(risk_ve_mask,risk_mask)],risks[risk_mask],ve[risk_ve_mask]*100,"Risk-compensation multiplier","Vaccine effectiveness (%)","C",diverging,[-150,-100,-50,-20,0,10,20,35,50,65,80,90])]
    for ax,(z,x,y,xl,yl,title,cmap,levels) in zip(axs,panels):
        # The risk panel has a hard biological reference boundary at a
        # multiplier of one.  Smoothing across that boundary falsely moves the
        # no-risk values; preserve the paired raw surface there.
        z=gaussian_filter(z,sigma=.65) if title != "C" else z
        cf=ax.contourf(x,y,z,levels=levels,cmap=cmap,extend="both")
        contour_levels=([-60,-40,-20,0,20,40,60,80] if title == "A" else ([0,10,20,40,60,80] if title == "B" else [-100,-50,-20,0,20,40,60,80]))
        cs=ax.contour(x,y,z,levels=contour_levels,colors="#334155",linewidths=.8); ax.clabel(cs,fmt="%d%%",fontsize=7)
        ax.set(xlabel=xl,ylabel=yl); ax.text(-.16,1.03,title,transform=ax.transAxes,fontweight="bold",fontsize=15)
        if title=="A": ax.scatter([30],[30],s=180,color="#0f172a",zorder=5); ax.scatter([70],[80],s=300,marker="*",color="#0f172a",zorder=5)
        if title=="B": ax.scatter([40],[45],s=300,marker="*",color="#0f172a",zorder=5)
        fig.colorbar(cf,ax=ax,orientation="horizontal",pad=.18,label="Median mortality reduction (%)")
    save(fig,render/"Figure_3_weighted_risk_review.png")
    # Figure 4: delay and immune onset
    delay=read(root/"fig4_rollout_delay_raw.csv"); gd=grouped(delay,("delay",))
    onset=read(a.fig4_onset_paired_raw or root/"fig4_immune_onset_raw.csv"); gi=grouped(onset,("immune_midpoint",))
    delays=np.array(sorted(k[0] for k in gd)); rd=np.array([reduction(gd[(d,)],base_c) for d in delays]); mids=np.array(sorted(k[0] for k in gi))
    ri=np.array([np.median([f(x,"mortality_reduction_pct") for x in gi[(d,)]]) if "mortality_reduction_pct" in onset[0] else reduction(gi[(d,)],base_c) for d in mids])
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.5),gridspec_kw={"wspace":.42})
    colors=["#2A9288","#D1C273","#CA6D4B"]
    labels=["At declaration\n(day 0)","Declaration\n+7 days","Declaration\n+14 days"]
    bars=axs[0].barh(labels,rd,color=colors,height=.56)
    axs[0].invert_yaxis(); axs[0].set(xlabel="Median mortality reduction (%)",title="50% community coverage")
    for b,v,c in zip(bars,rd,colors): axs[0].text(v+.8,b.get_y()+b.get_height()/2,f"{v:.0f}%",va="center",color=c,fontweight="bold")
    days=np.linspace(0,30,301)
    for midpoint,label,color in [(5,"Fast onset","#CA6D4B"),(10,"Standard onset","#2A9288"),(15,"Slow onset","#2878B5")]:
        axs[1].plot(days,45/(1+np.exp(-.5*(days-midpoint))),label=label,color=color,lw=2.8)
    axs[1].plot([0,10,10,30],[0,0,45,45],"--",color="#6B7280",lw=2,label="Step function")
    axs[1].set(xlim=(0,30),ylim=(0,50),xlabel="Days since vaccination",ylabel="Protection (%)")
    axs[1].legend(frameon=False,loc="lower right",fontsize=9)
    axs[2].plot(mids,ri,"o-",color="#2878B5",lw=2.8,ms=8)
    for x,y in zip(mids,ri): axs[2].annotate(f"{y:.1f}%",(x,y),xytext=(0,10),textcoords="offset points",ha="center",color="#2878B5",fontweight="bold")
    axs[2].set(title="Ring 2 vaccination\n(base operations vs base operations)",xlabel="Immune-onset midpoint (days)",ylabel="Median mortality reduction vs base ops (%)")
    axs[2].set_ylim(bottom=0)
    for i,ax in enumerate(axs):
        ax.text(-.18,1.03,"ABC"[i],transform=ax.transAxes,fontweight="bold",fontsize=15)
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color="#e5e7eb"); ax.set_axisbelow(True)
    save(fig,render/"Figure_4_template_review.png")
    # Figure 5 was exploratory and is omitted unless explicitly requested.
    if a.include_figure5:
        five=read(root/"fig5_rollout_grid_raw.csv"); g5=grouped(five,("delay","coverage")); ds=sorted({k[0] for k in g5}); cs=sorted({k[1] for k in g5});
        fig,ax=plt.subplots(figsize=(6.4,4.4))
        for d,col in zip(ds,["#21918C","#CC9E42","#CC6B4A"]): ax.plot(np.array(cs)*100,[reduction(g5[(d,c)],base_c) for c in cs],"o-",lw=2,label=f"{int(d)}-day delay",color=col)
        ax.set(title="Figure 5. Community vaccination timing",xlabel="Community vaccination coverage (%)",ylabel="Median mortality reduction (%)"); ax.legend(frameon=False); ax.grid(axis="y",color="#e5e7eb"); ax.spines[["top","right"]].set_visible(False); save(fig,render/"Figure_5_exploratory.png")
    # S3 delivery timing
    s3=read(root/"supp_s3_delivery_raw.csv"); gs=grouped(s3,("strategy",)); labels=[]; vals=[]
    for key in ("community40_base","ring2_enhanced"):
        rows=gs[(key,)]; counts=np.array([[f(r,"before_exposure"),f(r,"during_incubation"),f(r,"after_onset")] for r in rows]); denom=np.array([f(r,"n_vaccinated_cases") for r in rows]); vals.append(100*np.sum(counts,axis=0)/np.sum(denom)); labels.append("Community 40%" if key.startswith("community") else "Ring 2 + enhanced")
    fig,ax=plt.subplots(figsize=(7,3.8)); left=np.zeros(2); cols=["#21918C","#E8A12C","#CC6B4A"]
    for i,(name,col) in enumerate(zip(["Before exposure","During incubation","After onset"],cols)):
        x=np.array(vals)[:,i]; ax.barh(labels,x,left=left,color=col,label=name); left+=x
    ax.set(xlabel="Vaccinated eventual cases (%)",xlim=(0,100),title="Supplementary Figure S3. Timing of vaccination among eventual cases"); ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.35)); save(fig,render/"Supplementary_Figure_S3_current_review.png")
    # S4 independent VE
    s4=read(root/"supp_s4_independent_ve_raw.csv"); g4=grouped(s4,("ve_i","ve_m")); vis=np.array(sorted({k[0] for k in g4})); vms=np.array(sorted({k[1] for k in g4})); z4=np.array([[reduction(g4[(vi,vm)],base_c) for vi in vis] for vm in vms])
    fig,ax=plt.subplots(figsize=(5.8,4.5)); cf=ax.contourf(vis*100,vms*100,z4,levels=np.linspace(-20,95,13),cmap="RdYlBu",extend="both"); cs=ax.contour(vis*100,vms*100,z4,levels=[0,20,40,60,80],colors="#334155"); ax.clabel(cs,fmt="%d%%",fontsize=8); ax.set(title="Supplementary Figure S4. Independent vaccine effects",xlabel="Vaccine effectiveness against infection (%)",ylabel="Vaccine effectiveness against mortality (%)"); fig.colorbar(cf,ax=ax,label="Median mortality reduction (%)"); save(fig,render/"Supplementary_Figure_S4_current_review.png")
    with (render/"current_figure_grid_summary.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=["analysis","x","y","median_mortality_reduction_pct"]); w.writeheader();w.writerows(summary)

if __name__=="__main__": main()
