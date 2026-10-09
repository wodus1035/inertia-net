"""Gap figure: calibrated switching-probability curves for both cohorts over the
shared score, and the horizontal distance between their c50 (the switching-threshold
gap). 15-model mean score on held-out test, unweighted logistic fit per cohort."""
import sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, "figures"), os.path.join(_R, "scripts"), _R]
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
import figure_style as fs
from common import DOMAINS, REPO
fs.apply_style()
z=np.load(REPO/"results"/"scores15_test.npz", allow_pickle=True)
S={d:(z[f"{d}_s"].mean(0) if z[f"{d}_s"].ndim==2 else z[f"{d}_s"]) for d in DOMAINS}; Y={d:z[f"{d}_Y"] for d in DOMAINS}
pooled=np.r_[S["knu"],S["mimic"]]; lo,hi=np.percentile(pooled,[1,99]); grid=np.linspace(lo,hi,300)
fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),sharey=True)
for h,ax in enumerate(axes):
    c50={}
    for d in DOMAINS:
        r=sm.Logit(Y[d][:,h], sm.add_constant(S[d])).fit(disp=0); b0,b1=r.params; c50[d]=-b0/b1
        ax.plot(grid, 1/(1+np.exp(-(b0+b1*grid))), color=fs.COLOR[d], lw=fs.LW_DATA, label=fs.LABEL[d])
    gap=c50["knu"]-c50["mimic"]
    ax.axhline(0.5, color=fs.INK_MUTED, lw=fs.LW_CHROME, ls="--")
    for d in DOMAINS: ax.plot([c50[d],c50[d]],[0,0.5], color=fs.COLOR[d], lw=fs.LW_CHROME, ls=":")
    ax.annotate("", xy=(c50["knu"],0.5), xytext=(c50["mimic"],0.5), arrowprops=dict(arrowstyle="<->", color=fs.INK, lw=1.0, shrinkA=0, shrinkB=0))
    fs.clean_axes(ax,"y"); ax.set_ylim(0,1); ax.set_xlim(lo,hi); ax.set_title(f"Switch within {h+1} day{'s' if h else ''}   gap = {gap:.2f}", loc="left", fontsize=8.5)
    ax.set_xlabel("Readiness score $s(z)$", fontsize=8.5); ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("Switching probability", fontsize=8.5); axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(w_pad=1.0)
out=REPO.parent/"paper"/"Figures"/"figure4"; fig.savefig(str(out)+".pdf"); fig.savefig(str(out)+".png",dpi=130); fig.savefig(str(out)+".svg"); print("saved", out)
