"""Table S8 (calibration) and the threshold-rule sensitivity table over the 15 models.

Each model fits Platt (a,b) per cohort and horizon on its own fold's validation set and
applies it to the held-out test set, giving Brier/ECE/slope/intercept and, for five
threshold rules, the KNU-MIMIC threshold gap mapped back to score units.
"""
import numpy as np, statsmodels.api as sm
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve, roc_curve
from common import CV_SEEDS, DOMAINS, FOLDS, HORIZONS, ckpt_path, load_ckpt, rule, score_split
from src.data import load_domain, masks_of
from src.train_inertia_net import _make_cv_fold_masks

D = {d: load_domain(d) for d in DOMAINS}
comb = {d: masks_of(D[d], "train") | masks_of(D[d], "val") for d in DOMAINS}
test = {d: masks_of(D[d], "test") for d in DOMAINS}
folds = {d: _make_cv_fold_masks(comb[d], D[d]["eps"], 5, 42) for d in DOMAINS}
def platt(s, y):
    r = sm.Logit(y, sm.add_constant(s)).fit(disp=0); return r.params[1], r.params[0]      # a, b : logit p = a s + b
def ece(p, y, nb=10):
    e = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(p, e[1:-1]), 0, nb - 1); out = 0
    for i in range(nb):
        m = idx == i
        if m.any(): out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out
def cal_slope(p, y):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    r = sm.Logit(y, sm.add_constant(lg)).fit(disp=0); return r.params[1], r.params[0]
def thr_rules(pv, yv):
    fpr, tpr, th = roc_curve(yv, pv); pr, rc, pth = precision_recall_curve(yv, pv)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return {"F1": pth[np.argmax(f1)], "Youden": th[np.argmax(tpr - fpr)],
            "Sens>=0.80": th[np.argmax(tpr >= 0.80)], "Spec>=0.80": th[np.where(1 - fpr >= 0.80)[0][-1]],
            "Spec>=0.90": th[np.where(1 - fpr >= 0.90)[0][-1]]}
CAL = {d: {h: [] for h in range(3)} for d in DOMAINS}; GAP = {r: {h: [] for h in range(3)} for r in ("F1", "Youden", "Sens>=0.80", "Spec>=0.80", "Spec>=0.90")}
for f in FOLDS:
    for sd in CV_SEEDS:
        enc, hd, ck = load_ckpt(ckpt_path(f"cv_f{f}_s{sd}"), "cuda:0"); thr_s = {d: {} for d in DOMAINS}
        for d in DOMAINS:
            r = score_split(None, d, None, "cuda:0", enc, hd, ck); s, Y = r["s"], r["Y"]
            va, te = folds[d][f], test[d]
            for h in range(3):
                a, b = platt(s[va], Y[va, h]); pv, pt = 1 / (1 + np.exp(-(a * s[va] + b))), 1 / (1 + np.exp(-(a * s[te] + b)))
                sl, ic = cal_slope(pt, Y[te, h])
                CAL[d][h].append((brier_score_loss(Y[te, h], pt), ece(pt, Y[te, h]), sl, ic, brier_score_loss(Y[va, h], pv)))
                for rn, pth in thr_rules(pv, Y[va, h]).items():
                    pth = np.clip(pth, 1e-6, 1 - 1e-6); thr_s[d].setdefault(h, {})[rn] = (np.log(pth / (1 - pth)) - b) / a
        for h in range(3):
            for rn in GAP: GAP[rn][h].append(thr_s["knu"][h][rn] - thr_s["mimic"][h][rn])
        print(f"  cv_f{f}_s{sd}", flush=True)
rule("Table S8 - calibration of Platt probabilities, held-out test (15 models, mean +/- SD)")
print(f"  {'cohort':8s} {'h':>2s} {'Brier':>14s} {'ECE':>14s} {'Slope':>12s} {'Intercept':>13s} {'Brier val/test':>15s}")
for d in DOMAINS:
    for h in range(3):
        a = np.array(CAL[d][h]); m, sdv = a.mean(0), a.std(0)
        print(f"  {d.upper():8s} {h+1:>2d} {m[0]:.3f} ±{sdv[0]:.3f}   {m[1]:.3f} ±{sdv[1]:.3f}   {m[2]:.2f} ±{sdv[2]:.2f}   {m[3]:+.2f} ±{sdv[3]:.2f}   {m[4]:.3f} / {m[0]:.3f}")
rule("KNU-MIMIC threshold gap per rule (score units, rule applied on val; 15 models, mean +/- SD)")
for rn in GAP:
    print(f"  {rn:12s} " + "  ".join(f"h{h+1} {np.mean(GAP[rn][h]):+.3f} ±{np.std(GAP[rn][h]):.3f}" for h in range(3)))
