"""Supplementary figure: cohort selection flow chart; counts match Supplementary
Table S1. Output: paper/Figures/Sup/FigureS_cohort_flow.pdf/.png
"""
import sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, "figures"), os.path.join(_R, "scripts"), _R]
from pathlib import Path
import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figure_style as fs
fs.apply_style()
OUT = str(Path(_R).parent / "paper" / "Figures" / "Sup" / "FigureS_cohort_flow")

STEPS = ["Episodes with $\\geq$1 systemic\nantibacterial order",
         "Both IV and oral\nantibacterial therapy",
         "IV stop $\\leq$ oral stop",
         "Physiological record\n$\\leq$60 days with $\\geq$1 anchor-day"]
EXCL  = ["Excluded: no oral or no IV therapy", "Excluded: IV stop $>$ oral stop", "Excluded: no physiological data,\nrecord $>$60 days, or no anchor-day"]
N = {"knu":   [20107, 11979, 10278, 9503],
     "mimic": [59266, 13970, 8790, 8712]}
FINAL = {"knu": ("9,503 episodes", "8,390 patients", "148,361 anchor-days"),
         "mimic": ("8,712 episodes", "7,560 patients", "60,242 anchor-days")}
UNIT = {"knu": "hospital admissions", "mimic": "ICU stays"}

fig, ax = plt.subplots(figsize=(7.2, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0.3, 10); ax.axis("off")
BW, BH, EW, EH = 3.0, 0.82, 3.4, 0.80
ys = [9.0, 7.1, 5.2, 3.3]; YF = 1.3
XC = {"knu": 1.75, "mimic": 8.25}; XE = 5.0
def box(x, y, w, h, text, fc, ec, fs_=8.2, weight="normal", color=fs.INK):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs_, color=color, weight=weight, linespacing=1.25)
def arrow(x0, y0, x1, y1, color=fs.INK_SECONDARY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color, shrinkA=0, shrinkB=0))

for d in ("knu", "mimic"):
    xc, col = XC[d], fs.COLOR[d]
    ax.text(xc, 9.8, f"{fs.LABEL[d]} ({UNIT[d]})", ha="center", va="center", fontsize=9.5, weight="bold", color=col)
    for i, (y, n) in enumerate(zip(ys, N[d])):
        box(xc, y, BW, BH, f"{STEPS[i]}\n$n$ = {n:,}", fc="white", ec=col)
        if i < len(ys) - 1: arrow(xc, y - BH/2, xc, ys[i+1] + BH/2)
    arrow(xc, ys[-1] - BH/2, xc, YF + 0.5)
    box(xc, YF, BW, 1.0, "Final cohort\n" + "\n".join(FINAL[d]), fc=col, ec=col, fs_=8.0, weight="bold", color="white")
# shared exclusion boxes down the middle, arrows in from both sides
for i in range(len(ys) - 1):
    ym = (ys[i] + ys[i+1]) / 2
    ek, em = N["knu"][i] - N["knu"][i+1], N["mimic"][i] - N["mimic"][i+1]
    box(XE, ym, EW, EH, f"{EXCL[i]}\nKNU {ek:,}   ·   MIMIC-IV {em:,}", fc=fs.GRID, ec=fs.INK_MUTED, fs_=7.2, color=fs.INK_SECONDARY)
    arrow(XC["knu"], ym, XE - EW/2, ym); arrow(XC["mimic"], ym, XE + EW/2, ym)
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
fig.savefig(OUT + ".png", dpi=130); fig.savefig(OUT + ".svg"); fs.save(fig, OUT + ".pdf")
