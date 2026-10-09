"""Pooled XGBoost / logistic baselines trained on both cohorts without a cohort indicator,
then the same score-conditional analysis as Inertia-Net (log-OR at median/p90, c50 gap).
Features, folds and test split follow baseline_classical_ml. Writes
results/pooled_baseline.json and results/pooled_baseline_scores.npz.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, c50
from src.data import load_domain, make_Y, masks_of
from src.train_inertia_net import _make_cv_fold_masks
from src.baseline_classical_ml import build_features
from src.config import patient_of
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from xgboost import XGBClassifier
ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0]); ap.add_argument("--models", nargs="+", default=["xgb", "lr"]); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args(); H = [1, 2, 3]
D = {d: load_domain(d) for d in ("knu", "mimic")}
folds = {d: _make_cv_fold_masks(masks_of(D[d], "train") | masks_of(D[d], "val"), D[d]["eps"], 5, 42) for d in D}
te = {d: masks_of(D[d], "test") for d in D}
Xte = {d: build_features(D[d], te[d]) for d in D}; Yte = {d: make_Y(D[d]["delta"][te[d]], H) for d in D}
pat = {d: patient_of(pd.Series(D[d]["eps"][te[d]]).astype(str)).to_numpy() for d in D}
def logor(s_k, y_k, s_m, y_m):
    s = np.r_[s_k, s_m]; y = np.r_[y_k, y_m]; dm = np.r_[np.zeros(len(s_k)), np.ones(len(s_m))]; g = np.r_[pat["knu"], pat["mimic"]]
    X = sm.add_constant(np.column_stack([s, dm, s * dm]))
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g})
    b, V = r.params, r.cov_params(); out = {}
    for nm, s0 in (("median", np.median(s)), ("p90", np.percentile(s, 90))):
        est = b[2] + b[3] * s0; se = float(np.sqrt(max(V[2, 2] + s0**2 * V[3, 3] + 2 * s0 * V[2, 3], 0)))
        out[nm] = (float(est), float(est - 1.96 * se), float(est + 1.96 * se))
    out["interaction_p"] = float(r.pvalues[3]); return out
res = {m: {"fold": [], "auroc": {"knu": [], "mimic": []}, "logor_median": [], "logor_p90": [], "gap": [], "gap_sd_units": [], "interaction_p": [], "overlap": [], "cohort_auc": []} for m in a.models}
SAVE = {}
for f in a.folds:
    Xtr, Ytr = [], []
    for d in D:
        tr = np.zeros_like(folds[d][0])
        for k in range(5):
            if k != f: tr |= folds[d][k]
        Xtr.append(build_features(D[d], tr)); Ytr.append(make_Y(D[d]["delta"][tr], H))
    Xtr = np.vstack(Xtr); Ytr = np.vstack(Ytr); print(f"fold {f}: pooled train {len(Xtr):,}", flush=True)
    for m in a.models:
        au = {"knu": [], "mimic": []}; lo_med, lo_p90, gaps, gaps_sd, ips, ovl, cauc = [], [], [], [], [], [], []
        if m == "lr":
            sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); Xte_s = {d: sc.transform(Xte[d]) for d in D}
        for hi, h in enumerate(H):
            y = Ytr[:, hi]; pw = float((1 - y.mean()) / y.mean())
            if m == "xgb":
                clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss", n_jobs=a.threads, random_state=42, verbosity=0).fit(Xtr, y)
                S = {d: clf.predict(Xte[d], output_margin=True) for d in D}
            else:
                clf = LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: pw}).fit(Xtr_s, y)
                S = {d: clf.decision_function(Xte_s[d]) for d in D}
            for d in D: au[d].append(float(roc_auc_score(Yte[d][:, hi], S[d])))
            lo = logor(S["knu"], Yte["knu"][:, hi], S["mimic"], Yte["mimic"][:, hi])
            lo_med.append(lo["median"]); lo_p90.append(lo["p90"]); ips.append(lo["interaction_p"])
            g = c50(S["knu"], Yte["knu"][:, hi]) - c50(S["mimic"], Yte["mimic"][:, hi]); gaps.append(float(g))
            gaps_sd.append(float(g / np.std(np.r_[S["knu"], S["mimic"]])))
            # how far the score separates the cohorts: P(s_M > s_K)
            o = float(roc_auc_score(np.r_[np.zeros(len(S["knu"])), np.ones(len(S["mimic"]))], np.r_[S["knu"], S["mimic"]])); ovl.append(o); cauc.append(o)
            SAVE[f"{m}_f{f}_h{h}_knu"] = S["knu"]; SAVE[f"{m}_f{f}_h{h}_mimic"] = S["mimic"]
            print(f"  {m} h{h}: AUROC KNU {au['knu'][-1]:.3f} MIMIC {au['mimic'][-1]:.3f} | logOR@median {lo['median'][0]:.2f} ({lo['median'][1]:.2f}–{lo['median'][2]:.2f}) @p90 {lo['p90'][0]:.2f} | gap {g:.3f} logit = {gaps_sd[-1]:.3f} SD | interaction p {lo['interaction_p']:.3f} | P(s_M>s_K) {o:.3f}", flush=True)
        R = res[m]; R["fold"].append(f); R["auroc"]["knu"].append(au["knu"]); R["auroc"]["mimic"].append(au["mimic"]); R["logor_median"].append(lo_med); R["logor_p90"].append(lo_p90); R["gap"].append(gaps); R["gap_sd_units"].append(gaps_sd); R["interaction_p"].append(ips); R["overlap"].append(ovl); R["cohort_auc"].append(cauc)
json.dump(res, open(f"{REPO}/results/pooled_baseline.json", "w"), indent=1); np.savez_compressed(f"{REPO}/results/pooled_baseline_scores.npz", **SAVE); print("saved results/pooled_baseline.json")
