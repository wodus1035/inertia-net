"""Table 2 (performance comparison, held-out test): every row from one place.

Baselines are read from results/baselines/*/report.json and results/knu_only_test.json;
Inertia-Net is scored here from the 15 CV checkpoints. Writes LaTeX rows to stdout.
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from common import CV_TAGS, DOMAINS, REPO, rule, score_models

def ms(v): v=np.asarray(v,float); return f"${v.mean():.3f} \\pm {v.std():.3f}$"
rows = {}
for m, key in (("LogReg","logreg_test"),("XGBoost","xgboost_test")):
    rows[m] = {d: json.load(open(f"{REPO}/results/baselines/classical_ml_{d}/report.json"))[key] for d in DOMAINS}
ko = json.load(open(f"{REPO}/results/knu_only_test.json"))["results"]
rows["KNU-only$^{\\dagger}$"] = ko
S, D = score_models(CV_TAGS, DOMAINS, split="test", verbose=False)
inn = {d: {"auroc": [[roc_auc_score(D[d]["Y"][:,h], s) for h in range(3)] for s in S[d]],
           "auprc": [[average_precision_score(D[d]["Y"][:,h], s) for h in range(3)] for s in S[d]]} for d in DOMAINS}
rows["\\textbf{Inertia-Net}"] = inn
rule("Table 2 - LaTeX rows (fold mean +/- SD, n=15 for Inertia-Net/KNU-only, n=5 for LogReg/XGBoost)")
for name, r in rows.items():
    print(f"\\multirow{{3}}{{*}}{{{name}}}")
    for h in range(3):
        ak, pk = np.array(r["knu"]["auroc"])[:,h], np.array(r["knu"]["auprc"])[:,h]
        am, pm = np.array(r["mimic"]["auroc"])[:,h], np.array(r["mimic"]["auprc"])[:,h]
        comb = (ak.mean()+am.mean())/2
        print(f"  & {h+1} day{'s' if h else ''} & {ms(ak)} & {ms(pk)} & {ms(am)} & {ms(pm)} & {comb:.3f} \\\\")
    print("\\addlinespace")
n_k = len(D["knu"]["Y"]); n_m = len(D["mimic"]["Y"])
print(f"\nfor the footnote: KNU {n_k:,} and MIMIC-IV {n_m:,} anchor-days")
