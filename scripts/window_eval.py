"""Observation-window sensitivity (24/48/72 h): held-out test AUROC, AUPRC and c50 gap over
the 15 models of one window. IVOS_INPUT_DAYS must be set before this script runs, since
config reads it at import. Writes results/window_eval_d{D}.json for render_lookback.py."""
import json, os, sys, glob
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from common import CV_SEEDS, DOMAINS, FOLDS, REPO, c50, load_ckpt, rule, score_split
D = int(os.environ["IVOS_INPUT_DAYS"]); SC = os.environ.get("IVOS_SCRATCH", "/tmp/inertia_scratch")
pref = "cv" if D == 3 else f"w{D}"
R = {d: {"auroc": [], "auprc": []} for d in DOMAINS}; G = []
for f in FOLDS:
    for sd in CV_SEEDS:
        tag = f"{pref}_f{f}_s{sd}"; p = REPO / "models" / tag / "inertia_net_model.pth"
        if not p.exists():
            g = glob.glob(f"{SC}/out_{tag}/inertia_net/*/inertia_net_model.pth"); p = g[0] if g else None
        if p is None: print(f"  {tag} missing"); continue
        enc, hd, ck = load_ckpt(p, "cuda:0"); S = {}
        for d in DOMAINS:
            r = score_split(None, d, "test", "cuda:0", enc, hd, ck); S[d] = r
            R[d]["auroc"].append([roc_auc_score(r["Y"][:, h], r["s"]) for h in range(3)])
            R[d]["auprc"].append([average_precision_score(r["Y"][:, h], r["s"]) for h in range(3)])
        G.append([c50(S["knu"]["s"], S["knu"]["Y"][:, h]) - c50(S["mimic"]["s"], S["mimic"]["Y"][:, h]) for h in range(3)])
        print(f"  {tag}", flush=True)
rule(f"observation window {24*D} h - {len(G)} models, held-out test")
for d in DOMAINS:
    a, pr = np.array(R[d]["auroc"]), np.array(R[d]["auprc"])
    print(f"  {d.upper():6s} AUROC " + " ".join(f"h{h+1} {a[:,h].mean():.3f}±{a[:,h].std():.3f}" for h in range(3)) + "   AUPRC " + " ".join(f"{pr[:,h].mean():.3f}±{pr[:,h].std():.3f}" for h in range(3)))
G = np.array(G); print("  gap   " + " ".join(f"h{h+1} {G[:,h].mean():.3f}±{G[:,h].std():.3f}" for h in range(3)))
json.dump({"input_days": D, "n_models": len(G), "results": R, "gap": G.tolist()}, open(REPO / "results" / f"window_eval_d{D}.json", "w"), indent=1)
