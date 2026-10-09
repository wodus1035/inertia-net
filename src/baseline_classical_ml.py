"""Classical ML baseline: logistic regression and XGBoost on mask-aware summary
features (last / mean / std / slope / observation density per channel), one model
per domain and per horizon, sharing train_inertia_net's 5-fold patient-level CV.

    python -m src.baseline_classical_ml --domain knu --n_folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

from .config import OUTPUTS
from .data import load_domain, make_Y, masks_of
from .train_inertia_net import _subsample_mask, _make_cv_fold_masks


def summarize_channel_group(X, M):
    """X, M: (N, T, C) raw values / observation mask. Returns (N, 5*C) features:
    for each channel c, features are [last_c, mean_c, std_c, slope_c, density_c]
    (concatenated across channels)."""
    N, T, C = X.shape
    Mf = M.astype(np.float64)
    # NaN * 0 is NaN, so unobserved positions must be zeroed before masking.
    Xf = np.nan_to_num(X.astype(np.float64), nan=0.0)
    count = Mf.sum(axis=1)                                    # (N, C)
    density = count / T                                        # (N, C)

    safe_count = np.clip(count, 1, None)
    mean = (Xf * Mf).sum(axis=1) / safe_count                  # (N, C)
    diff = (Xf - mean[:, None, :]) * Mf
    var = (diff ** 2).sum(axis=1) / safe_count
    std = np.sqrt(np.clip(var, 0, None))

    # last observed value per (N, C): reverse time, find first mask hit
    rev_mask = M[:, ::-1, :]
    any_obs = M.any(axis=1)                                    # (N, C)
    rev_first_idx = np.argmax(rev_mask, axis=1)                # (N, C), 0 if none observed
    last_t = (T - 1) - rev_first_idx
    n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing="ij")
    last_val = Xf[n_idx, last_t, c_idx]
    last_val = np.where(any_obs, last_val, 0.0)                 # 0 if channel never observed

    # slope via weighted least squares over observed timesteps (mask as weight)
    t_idx = np.arange(T, dtype=np.float64)[None, :, None]       # (1, T, 1)
    n = count
    sx = (Mf * t_idx).sum(axis=1)
    sxx = (Mf * t_idx ** 2).sum(axis=1)
    sy = (Xf * Mf).sum(axis=1)
    sxy = (Xf * Mf * t_idx).sum(axis=1)
    denom = n * sxx - sx ** 2
    slope = np.where((n >= 2) & (np.abs(denom) > 1e-8),
                      (n * sxy - sx * sy) / np.where(np.abs(denom) > 1e-8, denom, 1.0),
                      0.0)

    # interleave per channel: [last_0, mean_0, std_0, slope_0, density_0, last_1, ...]
    stacked = np.stack([last_val, mean, std, slope, density], axis=-1)  # (N, C, 5)
    return stacked.reshape(N, C * 5)


def build_features(d, idx):
    vit_feat = summarize_channel_group(d["vit"][idx], d["vmask"][idx])   # (N, 11*5)
    lab_feat = summarize_channel_group(d["lab"][idx], d["lmask"][idx])   # (N, 16*5)
    static_feat = np.nan_to_num(d["static"][idx].astype(np.float64), nan=0.0)  # (N, 2)
    return np.concatenate([vit_feat, lab_feat, static_feat], axis=1)


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    return float(thr[best]) if best < len(thr) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["knu", "mimic"], required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--subsample_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    # Folds run as threads (the domain arrays are large and both XGBoost and BLAS
    # release the GIL). fold_jobs * xgb_threads is the real core footprint.
    ap.add_argument("--fold_jobs", type=int, default=5)
    ap.add_argument("--xgb_threads", type=int, default=4)
    args = ap.parse_args()

    horizons = sorted(args.horizons)
    n_horizons = len(horizons)

    print(f"XGBoost available: {HAVE_XGB}" + ("" if HAVE_XGB else
          "  (install with: pip install xgboost --break-system-packages)"))

    d = load_domain(args.domain)
    tr_orig, va_orig, te = masks_of(d, "train"), masks_of(d, "val"), masks_of(d, "test")
    combined = tr_orig | va_orig
    folds = _make_cv_fold_masks(combined, d["eps"], args.n_folds, args.cv_seed)

    results_logreg = {"auroc": [], "auprc": [], "f1": []}
    results_xgb = {"auroc": [], "auprc": [], "f1": []}
    # Each fold model also predicts the untouched test split, matching Inertia-Net's
    # protocol. The F1 threshold comes from that fold's validation fold, never test.
    test_logreg = {"auroc": [], "auprc": [], "f1": []}
    test_xgb = {"auroc": [], "auprc": [], "f1": []}
    X_te_raw = build_features(d, te)
    Y_te = make_Y(d["delta"][te], horizons)
    print(f"held-out test: n={int(te.sum()):,}  positive rates="
          f"{np.round(Y_te.mean(0)*100, 2)}%")

    def run_fold(fold):
        va_mask = folds[fold]
        tr_mask = np.zeros_like(combined)
        for f in range(args.n_folds):
            if f != fold:
                tr_mask |= folds[f]
        tr_mask = _subsample_mask(tr_mask, args.subsample_frac, args.seed)

        X_tr, Y_tr = build_features(d, tr_mask), make_Y(d["delta"][tr_mask], horizons)
        X_va, Y_va = build_features(d, va_mask), make_Y(d["delta"][va_mask], horizons)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)
        X_te_s = scaler.transform(X_te_raw)

        fold_auroc_lr, fold_auprc_lr, fold_f1_lr = [], [], []
        fold_auroc_xgb, fold_auprc_xgb, fold_f1_xgb = [], [], []
        t_auroc_lr, t_auprc_lr, t_f1_lr = [], [], []
        t_auroc_xgb, t_auprc_xgb, t_f1_xgb = [], [], []

        for h_idx, h in enumerate(horizons):
            y_tr, y_va = Y_tr[:, h_idx], Y_va[:, h_idx]
            p = max(y_tr.mean(), 1e-6)
            pos_weight = (1 - p) / p

            clf_lr = LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: pos_weight})
            clf_lr.fit(X_tr_s, y_tr)
            prob_lr = clf_lr.predict_proba(X_va_s)[:, 1]
            thr_lr = best_f1_threshold(y_va, prob_lr)
            fold_auroc_lr.append(roc_auc_score(y_va, prob_lr) if len(np.unique(y_va)) > 1 else np.nan)
            fold_auprc_lr.append(average_precision_score(y_va, prob_lr) if len(np.unique(y_va)) > 1 else np.nan)
            fold_f1_lr.append(f1_score(y_va, (prob_lr >= thr_lr).astype(int), zero_division=0))

            y_te = Y_te[:, h_idx]
            prob_lr_te = clf_lr.predict_proba(X_te_s)[:, 1]
            t_auroc_lr.append(roc_auc_score(y_te, prob_lr_te) if len(np.unique(y_te)) > 1 else np.nan)
            t_auprc_lr.append(average_precision_score(y_te, prob_lr_te) if len(np.unique(y_te)) > 1 else np.nan)
            t_f1_lr.append(f1_score(y_te, (prob_lr_te >= thr_lr).astype(int), zero_division=0))

            if HAVE_XGB:
                clf_xgb = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=pos_weight, eval_metric="logloss",
                    subsample=0.8, colsample_bytree=0.8, n_jobs=args.xgb_threads,
                )
                clf_xgb.fit(X_tr, y_tr)
                prob_xgb = clf_xgb.predict_proba(X_va)[:, 1]
                thr_xgb = best_f1_threshold(y_va, prob_xgb)
                fold_auroc_xgb.append(roc_auc_score(y_va, prob_xgb) if len(np.unique(y_va)) > 1 else np.nan)
                fold_auprc_xgb.append(average_precision_score(y_va, prob_xgb) if len(np.unique(y_va)) > 1 else np.nan)
                fold_f1_xgb.append(f1_score(y_va, (prob_xgb >= thr_xgb).astype(int), zero_division=0))

                prob_xgb_te = clf_xgb.predict_proba(X_te_raw)[:, 1]
                t_auroc_xgb.append(roc_auc_score(y_te, prob_xgb_te) if len(np.unique(y_te)) > 1 else np.nan)
                t_auprc_xgb.append(average_precision_score(y_te, prob_xgb_te) if len(np.unique(y_te)) > 1 else np.nan)
                t_f1_xgb.append(f1_score(y_te, (prob_xgb_te >= thr_xgb).astype(int), zero_division=0))

        print(f"[{args.domain} fold {fold}] LogReg  cv={np.round(fold_auroc_lr,4)}  test={np.round(t_auroc_lr,4)}",
              flush=True)
        if HAVE_XGB:
            print(f"[{args.domain} fold {fold}] XGBoost cv={np.round(fold_auroc_xgb,4)}  test={np.round(t_auroc_xgb,4)}",
                  flush=True)
        return {"fold": fold,
                "cv_lr": (fold_auroc_lr, fold_auprc_lr, fold_f1_lr),
                "te_lr": (t_auroc_lr, t_auprc_lr, t_f1_lr),
                "cv_xgb": (fold_auroc_xgb, fold_auprc_xgb, fold_f1_xgb),
                "te_xgb": (t_auroc_xgb, t_auprc_xgb, t_f1_xgb)}

    print(f"running {args.n_folds} folds with fold_jobs={args.fold_jobs}, "
          f"xgb_threads={args.xgb_threads}", flush=True)
    fold_out = Parallel(n_jobs=args.fold_jobs, prefer="threads")(
        delayed(run_fold)(f) for f in range(args.n_folds))
    for r in sorted(fold_out, key=lambda r: r["fold"]):        # fold order, not finish order
        for store, key in [(results_logreg, "cv_lr"), (test_logreg, "te_lr"),
                           (results_xgb, "cv_xgb"), (test_xgb, "te_xgb")]:
            if key.endswith("xgb") and not HAVE_XGB:
                continue
            a, b, c = r[key]
            store["auroc"].append(a); store["auprc"].append(b); store["f1"].append(c)

    print(f"\n{'='*70}\n{args.domain.upper()} — CLASSICAL ML BASELINE, {args.n_folds}-fold CV (mean \u00b1 SD)\n{'='*70}")

    def _report(name, res):
        arr_auroc, arr_auprc, arr_f1 = np.array(res["auroc"]), np.array(res["auprc"]), np.array(res["f1"])
        print(f"\n{name}:")
        for h_idx, h in enumerate(horizons):
            print(f"  horizon={h}:  AUROC={arr_auroc[:,h_idx].mean():.4f}\u00b1{arr_auroc[:,h_idx].std():.4f}  "
                  f"AUPRC={arr_auprc[:,h_idx].mean():.4f}\u00b1{arr_auprc[:,h_idx].std():.4f}  "
                  f"F1={arr_f1[:,h_idx].mean():.4f}\u00b1{arr_f1[:,h_idx].std():.4f}")
        return arr_auroc, arr_auprc, arr_f1

    lr_auroc, lr_auprc, lr_f1 = _report("Logistic Regression", results_logreg)
    if HAVE_XGB:
        xgb_auroc, xgb_auprc, xgb_f1 = _report("XGBoost", results_xgb)

    print(f"\n{'='*70}\n{args.domain.upper()} — SAME MODELS ON THE HELD-OUT TEST SPLIT "
          f"(mean \u00b1 SD over {args.n_folds} fold models)\n{'='*70}")
    lr_t = _report("Logistic Regression (test)", test_logreg)
    if HAVE_XGB:
        xgb_t = _report("XGBoost (test)", test_xgb)

    out_dir = Path(OUTPUTS) / "baselines" / f"classical_ml_{args.domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "domain": args.domain, "horizons": horizons, "n_folds": args.n_folds,
        "logreg": {"auroc": lr_auroc.tolist(), "auprc": lr_auprc.tolist(), "f1": lr_f1.tolist()},
        "eval_note": "top-level logreg/xgboost blocks are CV-fold estimates; "
                     "*_test blocks are the same fold models evaluated on the "
                     "untouched held-out test split (Inertia-Net's protocol)",
        "n_test_samples": int(te.sum()),
        "logreg_test": {"auroc": lr_t[0].tolist(), "auprc": lr_t[1].tolist(), "f1": lr_t[2].tolist()},
    }
    if HAVE_XGB:
        report["xgboost"] = {"auroc": xgb_auroc.tolist(), "auprc": xgb_auprc.tolist(), "f1": xgb_f1.tolist()}
        report["xgboost_test"] = {"auroc": xgb_t[0].tolist(), "auprc": xgb_t[1].tolist(), "f1": xgb_t[2].tolist()}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out_dir/'report.json'}")


if __name__ == "__main__":
    main()