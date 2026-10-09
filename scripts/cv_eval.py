"""Held-out test evaluation of the final specification (5 folds x 3 seeds = 15 models).

Prints discrimination, the site gap in c50, and a domain probe on s(z) alone.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CV_TAGS, DOMAINS, c50, mean_sd, rule, score_models

S, D = score_models(CV_TAGS, DOMAINS, split="test")

rule("held-out test - ds_nocap (5 folds x 3 seeds = 15 models)")
print(f"  {'':8s} {'n':>8s} {'prev':>7s} " +
      " ".join(f"{'h' + str(h + 1) + ' AUROC':>16s}" for h in range(3)))
for dom in DOMAINS:
    Y = D[dom]["Y"]
    cells = [mean_sd([roc_auc_score(Y[:, h], s) for s in S[dom]]) for h in range(3)]
    print(f"  {dom.upper():8s} {len(Y):>8,} {Y[:, 0].mean():>7.1%} " +
          " ".join(f"{c:>16s}" for c in cells))

rule("gap = c50(KNU) - c50(MIMIC)   [unweighted post-hoc logistic, observed 50% point]")
for h in range(3):
    ck = np.array([c50(S["knu"][i], D["knu"]["Y"][:, h]) for i in range(len(CV_TAGS))])
    cm = np.array([c50(S["mimic"][i], D["mimic"]["Y"][:, h]) for i in range(len(CV_TAGS))])
    g = ck - cm
    print(f"  h{h + 1}  c50 KNU {ck.mean():>6.3f}  MIMIC {cm.mean():>6.3f}   "
          f"gap {g.mean():>6.3f} ±{g.std():.3f}")

rule("domain probe - s(z) alone (patient-level half split)")
pt = np.r_[D["knu"]["pat"], D["mimic"]["pat"]]
dm = np.r_[np.zeros(len(D["knu"]["pat"])), np.ones(len(D["mimic"]["pat"]))]
up = np.array(sorted(set(pt))); np.random.RandomState(0).shuffle(up)
fit = np.isin(pt, up[:len(up) // 2])
for i, sd in enumerate(CV_TAGS):
    s_ = np.r_[S["knu"][i], S["mimic"][i]].reshape(-1, 1)
    sc = StandardScaler().fit(s_[fit])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(s_[fit]), dm[fit])
    auc = roc_auc_score(dm[~fit], lr.predict_proba(sc.transform(s_[~fit]))[:, 1])
    print(f"  {sd:10s}  AUC {auc:.4f}")
