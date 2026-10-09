"""Per-cohort vs shared cutpoints (fold 0, 3 seeds): per-cohort AUROC, domain separability
of s, score-conditional cohort log-OR at the pooled median, and the c50 gap.
Writes results/cutpoint_ablation.csv.

usage: python cutpoint_ablation.py --gpu 0 [--tags_a cv_f0_s42 ...] [--tags_b sc_f0_s42 ...]
"""
import argparse, numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from common import ckpt_path, load_ckpt, score_split, rule
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--tags_a", nargs="+", default=["cv_f0_s42", "cv_f0_s43", "cv_f0_s44"])
ap.add_argument("--tags_b", nargs="*", default=["sc_f0_s42", "sc_f0_s43", "sc_f0_s44"])
ap.add_argument("--tags_c", nargs="*", default=["nm_f0_s42", "nm_f0_s43", "nm_f0_s44"])
ap.add_argument("--tags_d", nargs="*", default=["sn_f0_s42", "sn_f0_s43", "sn_f0_s44"])
args = ap.parse_args(); dev = f"cuda:{args.gpu}"

def c50(s, y):
    r = sm.Logit(y, sm.add_constant(s)).fit(disp=0); return -r.params[0] / r.params[1], r.params[1]
def logor_median(sk, yk, pk, sm_, ym, pm):
    s = np.r_[sk, sm_]; y = np.r_[yk, ym]; d = np.r_[np.zeros(len(sk)), np.ones(len(sm_))]; g = np.r_[pk, pm]
    s0 = np.median(s); X = sm.add_constant(np.column_stack([s - s0, d, (s - s0) * d]))
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g})
    return r.params[2], r.bse[2]
def evaluate(tag):
    enc, hd, ck = load_ckpt(ckpt_path(tag), dev)
    R = {d: score_split(None, d, "test", dev, enc, hd, ck) for d in ("knu", "mimic")}
    sk, sm_ = R["knu"]["s"], R["mimic"]["s"]; out = {"tag": tag, "shared": bool(ck["hp"].get("shared_cutpoints", False)), "lambda_mmd": float(ck["hp"].get("lambda_mmd", float("nan")))}
    for h in range(3):
        out[f"auroc_knu_h{h+1}"] = roc_auc_score(R["knu"]["Y"][:, h], sk); out[f"auroc_mimic_h{h+1}"] = roc_auc_score(R["mimic"]["Y"][:, h], sm_)
    lab = np.r_[np.zeros(len(sk)), np.ones(len(sm_))]; out["domain_auroc"] = roc_auc_score(lab, np.r_[sk, sm_])   # = P(s_M > s_K)
    out["mean_s_knu"], out["mean_s_mimic"] = sk.mean(), sm_.mean()
    for h in range(3):
        b, se = logor_median(sk, R["knu"]["Y"][:, h], R["knu"]["pat"], sm_, R["mimic"]["Y"][:, h], R["mimic"]["pat"]); out[f"logor_h{h+1}"] = b
        ck_, ak = c50(sk, R["knu"]["Y"][:, h]); cm_, am = c50(sm_, R["mimic"]["Y"][:, h]); out[f"gap_h{h+1}"] = ck_ - cm_; out[f"slope_knu_h{h+1}"], out[f"slope_mimic_h{h+1}"] = ak, am
    cuts = {d: hd.cutpoints(d).detach().cpu().numpy().round(3).tolist() for d in ("knu", "mimic")}; out["cuts"] = cuts
    print(f"  {tag}: shared={out['shared']} domainAUC={out['domain_auroc']:.3f} AUROC K={out['auroc_knu_h1']:.3f} M={out['auroc_mimic_h1']:.3f} logOR h1={out['logor_h1']:.2f} gap h1={out['gap_h1']:.3f} cuts={cuts}", flush=True)
    return out
import os
rows = [evaluate(t) for t in args.tags_a + args.tags_b + args.tags_c + args.tags_d if os.path.exists(ckpt_path(t))]
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
rule("fold 0 x 3 seeds - cutpoint x MMD 2x2; mean +/- SD")
cols = ["auroc_knu_h1", "auroc_mimic_h1", "auroc_knu_h3", "auroc_mimic_h3", "domain_auroc", "mean_s_knu", "mean_s_mimic", "logor_h1", "logor_h2", "logor_h3", "gap_h1", "gap_h2", "gap_h3", "slope_knu_h1", "slope_mimic_h1"]
groups = (("A per-domain, MMD", df[(~df["shared"]) & (df["lambda_mmd"] > 0)]), ("B shared, MMD", df[(df["shared"]) & (df["lambda_mmd"] > 0)]),
          ("C per-domain, no MMD", df[(~df["shared"]) & (df["lambda_mmd"] == 0)]), ("D shared, no MMD", df[(df["shared"]) & (df["lambda_mmd"] == 0)]))
for name, grp in groups:
    if len(grp): print(f"  {name:22s} " + "  ".join(f"{c}={grp[c].mean():.3f}±{grp[c].std(ddof=0):.3f}" for c in cols))
df.to_csv(f"{__import__('common').REPO}/results/cutpoint_ablation.csv", index=False); print("saved results/cutpoint_ablation.csv")
