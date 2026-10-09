"""Adjusted cohort effect from pooled models carrying a cohort indicator: the LR cohort
coefficient (log-OR) and the mean/median counterfactual logit shift from flipping the
cohort feature in pooled XGB. Writes results/pooled_indicator.json.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO
from src.data import load_domain, make_Y, masks_of
from src.train_inertia_net import _make_cv_fold_masks
from src.baseline_classical_ml import build_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0]); ap.add_argument("--threads", type=int, default=16); a = ap.parse_args(); H = [1, 2, 3]
D = {d: load_domain(d) for d in ("knu", "mimic")}
folds = {d: _make_cv_fold_masks(masks_of(D[d], "train") | masks_of(D[d], "val"), D[d]["eps"], 5, 42) for d in D}
te = {d: masks_of(D[d], "test") for d in D}
Xte = {d: build_features(D[d], te[d]) for d in D}; Yte = {d: make_Y(D[d]["delta"][te[d]], H) for d in D}
res = {"fold": [], "lr_cohort_logor": [], "xgb_cf_mean": [], "xgb_cf_median": [], "xgb_auroc": {"knu": [], "mimic": []}}
for f in a.folds:
    Xtr, Ytr, Dtr = [], [], []
    for d in D:
        tr = np.zeros_like(folds[d][0])
        for k in range(5):
            if k != f: tr |= folds[d][k]
        X = build_features(D[d], tr); Xtr.append(X); Ytr.append(make_Y(D[d]["delta"][tr], H)); Dtr.append(np.full(len(X), 1.0 if d == "mimic" else 0.0))
    Xtr = np.vstack(Xtr); Ytr = np.vstack(Ytr); Dtr = np.concatenate(Dtr); Xtr_i = np.column_stack([Xtr, Dtr])
    Xte_all = np.vstack([Xte["knu"], Xte["mimic"]]); n_k = len(Xte["knu"])
    sc = StandardScaler().fit(Xtr); Xtr_s = np.column_stack([sc.transform(Xtr), Dtr]); Xte_s = sc.transform(Xte_all)
    lr_b, cf_mean, cf_med = [], [], []; au = {"knu": [], "mimic": []}
    for hi, h in enumerate(H):
        y = Ytr[:, hi]; pw = float((1 - y.mean()) / y.mean())
        lr = LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: pw}).fit(Xtr_s, y); lr_b.append(float(lr.coef_[0, -1]))
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss", n_jobs=a.threads, random_state=42, verbosity=0).fit(Xtr_i, y)
        m1 = clf.predict(np.column_stack([Xte_all, np.ones(len(Xte_all))]), output_margin=True); m0 = clf.predict(np.column_stack([Xte_all, np.zeros(len(Xte_all))]), output_margin=True)
        diff = m1 - m0; cf_mean.append(float(diff.mean())); cf_med.append(float(np.median(diff)))
        real = clf.predict(np.column_stack([Xte_all, np.r_[np.zeros(n_k), np.ones(len(Xte_all) - n_k)]]), output_margin=True)
        au["knu"].append(float(roc_auc_score(Yte["knu"][:, hi], real[:n_k]))); au["mimic"].append(float(roc_auc_score(Yte["mimic"][:, hi], real[n_k:])))
        print(f"fold {f} h{h}: LR cohort log-OR {lr_b[-1]:.2f} | XGB counterfactual logit(MIMIC)−logit(KNU) mean {cf_mean[-1]:.2f} median {cf_med[-1]:.2f} | XGB+ind AUROC KNU {au['knu'][-1]:.3f} MIMIC {au['mimic'][-1]:.3f}", flush=True)
    res["fold"].append(f); res["lr_cohort_logor"].append(lr_b); res["xgb_cf_mean"].append(cf_mean); res["xgb_cf_median"].append(cf_med); res["xgb_auroc"]["knu"].append(au["knu"]); res["xgb_auroc"]["mimic"].append(au["mimic"])
json.dump(res, open(f"{REPO}/results/pooled_indicator.json", "w"), indent=1); print("saved")
