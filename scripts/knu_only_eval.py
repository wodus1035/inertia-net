"""KNU-only row of Table 2: the same architecture trained on KNU alone and applied to MIMIC
zero-shot, using KNU normalisation statistics for both cohorts (MIMIC statistics would leak).
Writes results/knu_only_test.json, which table2.py reads.
"""
import os
import glob, json, re
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from common import CV_SEEDS, FOLDS, HORIZONS, REPO, ckpt_path, load_ckpt, mean_sd, rule, score
from src.data import load_domain, masks_of, make_Y

SC = os.environ.get("IVOS_SCRATCH", "/tmp/inertia_scratch")
tags = [f"ko_f{f}_s{s}" for f in FOLDS for s in CV_SEEDS]
paths = []
for t in tags:
    p = ckpt_path(t)
    if not p.exists():
        g = glob.glob(f"{SC}/out_{t}/inertia_net/*/inertia_net_model.pth"); p = g[0] if g else None
    paths.append((t, p))
paths = [(t, p) for t, p in paths if p]
print(f"KNU-only checkpoints {len(paths)}/15")
D = {d: load_domain(d) for d in ("knu", "mimic")}
idx = {d: np.where(masks_of(D[d], "test"))[0] for d in D}
Y = {d: make_Y(D[d]["delta"][idx[d]], HORIZONS) for d in D}
R = {d: {"auroc": [], "auprc": []} for d in D}
for t, p in paths:
    enc, hd, ck = load_ckpt(p, "cuda:0"); st_k = ck["norm_stats_per_domain"]["knu"]
    for d in D:
        i = idx[d]; dd = D[d]
        s, _ = score(enc, hd, dd["vit"][i], dd["vmask"][i], dd["lab"][i], dd["lmask"][i], dd["static"][i], st_k, "cuda:0")
        R[d]["auroc"].append([roc_auc_score(Y[d][:, h], s) for h in range(3)])
        R[d]["auprc"].append([average_precision_score(Y[d][:, h], s) for h in range(3)])
    print(f"  {t}", flush=True)
rule("KNU-only - held-out test, MIMIC zero-shot with KNU normalisation statistics")
print(f"  {'':4s} {'KNU AUROC':>16s} {'KNU AUPRC':>16s} {'MIMIC AUROC':>16s} {'MIMIC AUPRC':>16s}")
for h in range(3):
    print(f"  h{h+1}  " + " ".join(f"{mean_sd(np.array(R[d][m])[:, h]):>16s}" for d in ("knu", "mimic") for m in ("auroc", "auprc")))
json.dump({"tags": [t for t, _ in paths], "n_models": len(paths), "results": R}, open(f"{REPO}/results/knu_only_test.json", "w"), indent=1)
print(f"-> {REPO}/results/knu_only_test.json")
