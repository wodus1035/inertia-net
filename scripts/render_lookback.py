"""Supplementary tab:lookback - LaTeX rows for 24/48/72 h x 3 horizons x KNU/MIMIC
AUROC/AUPRC, plus the c50 gap per window. Reads results/window_eval_d{1,2,3}.json and
bolds the best value in each column."""
import json, numpy as np
from common import REPO
J={d:json.load(open(REPO/"results"/f"window_eval_d{d}.json")) for d in (1,2,3) if (REPO/"results"/f"window_eval_d{d}.json").exists()}
def ms(v): v=np.asarray(v,float); return f"${v.mean():.3f} \\pm {v.std():.3f}$"
best={}
for h in range(3):
    for k,(dom,met) in enumerate((("knu","auroc"),("knu","auprc"),("mimic","auroc"),("mimic","auprc"))):
        best[(h,k)]=max(np.array(J[d]["results"][dom][met])[:,h].mean() for d in J)
for d in sorted(J):
    print(f"\\multirow{{3}}{{*}}{{{24*d}\\,h}}")
    for h in range(3):
        cells=[]
        for k,(dom,met) in enumerate((("knu","auroc"),("knu","auprc"),("mimic","auroc"),("mimic","auprc"))):
            v=np.array(J[d]["results"][dom][met])[:,h]; c=ms(v)
            if abs(v.mean()-best[(h,k)])<5e-4: c=c.replace("$","",1).replace("$","",1); c=f"$\\mathbf{{{c}}}$"
            cells.append(c)
        print(f"  & {h+1} day{'s' if h else ' '} & " + " & ".join(cells) + " \\\\")
    print("\\addlinespace")
print("\n% gap (c50, score units) by window:")
for d in sorted(J):
    g=np.array(J[d]["gap"]); print(f"%  {24*d} h: " + "  ".join(f"h{h+1} {g[:,h].mean():.3f}±{g[:,h].std():.3f}" for h in range(3)) + f"   (n={J[d]['n_models']})")
