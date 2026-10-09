"""Figure 6: permutation importance, KNU vs MIMIC-IV, vitals and labs separated.
usage: python plot_importance_main.py <npz> <out_pdf>"""
import sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, "figures"), os.path.join(_R, "scripts"), _R]
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figure_style as fs
from common import DOMAINS
fs.apply_style(); z=np.load(sys.argv[1], allow_pickle=True); NAMES=[str(x) for x in z["names"]]
LABEL={"SBP":"Systolic BP","DBP":"Diastolic BP","Pulse":"Heart rate","Resp_Rate":"Respiratory rate","Body_Temp":"Body temperature","SpO2":"SpO$_2$","resp_support":"Respiratory support","FiO2":"FiO$_2$","O2_Amount":"Oxygen amount","Flow_Rate":"Oxygen flow rate","AVPU":"AVPU",
       "WBC":"WBC","Platelets":"Platelets","Hemoglobin":"Hemoglobin","Lymphocytes":"Lymphocytes","Neutrophils":"Neutrophils","Creatinine":"Creatinine","BUN":"BUN","Sodium":"Sodium","Potassium":"Potassium","pH":"pH","pCO2":"pCO$_2$","pO2":"pO$_2$","HCO3":"HCO$_3$","Total_CO2":"Total CO$_2$","O2_Sat":"O$_2$ saturation"}
VIT=["SBP","DBP","Pulse","Resp_Rate","Body_Temp","SpO2","resp_support","FiO2","O2_Amount","Flow_Rate","AVPU"]
M={d:z[d] for d in DOMAINS}                       # (n_models, n_channels)
vit_idx=[NAMES.index(v) for v in VIT]; lab_idx=[i for i in range(len(NAMES)) if NAMES[i] not in VIT]
rows=[(LABEL.get(NAMES[i],NAMES[i]), {d:M[d][:,i] for d in DOMAINS}) for i in vit_idx]
LIGHT = {"knu": "#E08A94", "mimic": "#7FA9D6"}
rows.sort(key=lambda r: -abs(r[1]["knu"].mean()-r[1]["mimic"].mean()))   # largest cohort difference first
rows.append(("All laboratory channels", {d:M[d][:,lab_idx].sum(1) for d in DOMAINS}))
n=len(rows); ypos=np.arange(n)[::-1].astype(float); ypos[-1]-=0.8   # extra gap before the lab-sum row
fig,ax=plt.subplots(figsize=(6.2,3.9)); W=0.36; OFF={"knu":+W/2,"mimic":-W/2}
for d in DOMAINS:
    ax.barh(ypos+OFF[d], [v[d].mean() for _,v in rows], height=W, xerr=[1.96*v[d].std(ddof=1)/np.sqrt(len(v[d])) for _,v in rows], color=LIGHT[d], edgecolor=fs.SURFACE, linewidth=0.5, label=fs.LABEL[d], error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor=fs.INK_SECONDARY), zorder=3)
ax.set_yticks(ypos); ax.set_yticklabels([lab for lab,_ in rows], fontsize=8)
ax.axvline(0,color=fs.INK_MUTED,lw=fs.LW_CHROME); ax.axhline((ypos[-1]+ypos[-2])/2,color=fs.GRID,lw=1.0); fs.clean_axes(ax,"x"); ax.tick_params(axis="x",labelsize=7.5)
ax.set_xlabel("Permutation importance", fontsize=8.5)
ax.legend(frameon=False, loc="center right", fontsize=8); ax.set_ylim(ypos[-1]-0.7, ypos[0]+0.7)
fig.tight_layout(); out=sys.argv[2]; fig.savefig(out); fig.savefig(out.replace(".pdf",".png"),dpi=130); fig.savefig(out.replace(".pdf",".svg")); print("saved",out,f"(n models = {z['knu'].shape[0]})")
