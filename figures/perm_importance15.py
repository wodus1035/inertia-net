"""Per-channel permutation importance by cohort over the 15 CV models (5 folds x 3
seeds). On the held-out test set, channel c's (value, mask) pair is shuffled across
anchors and the drop in 1-day AUROC is divided by (baseline - 0.5)."""
import sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, "figures"), os.path.join(_R, "scripts"), _R]
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import figure_style as fs
from common import CV_TAGS, DATA, DOMAINS, FOLDS, REPO, ckpt_path, load_ckpt, score
from src.data import load_domain, masks_of, make_Y
OUT = REPO.parent / "paper" / "Figures" / "Sup"; fs.apply_style()
def _save_both(fig, path, dpi=600):
    from pathlib import Path as _P
    fig.savefig(path, dpi=dpi); fig.savefig(_P(str(path)).with_suffix('.png'), dpi=130); plt.close(fig); print(f'  saved -> {path} (+png)')
fs.save = _save_both; rng = np.random.default_rng(0)
z = np.load(f"{DATA}/knu_tensors_d3_b36x2h.npz", allow_pickle=True); VC = [str(c) for c in z["vit_cols"]]; LC = [str(c) for c in z["lab_cols"]]
NAMES = VC + LC
D = {d: load_domain(d) for d in DOMAINS}; te = {d: np.where(masks_of(D[d], "test"))[0] for d in DOMAINS}
IMP = {d: [] for d in DOMAINS}
for tag in CV_TAGS:
    enc, hd, ck = load_ckpt(ckpt_path(tag), "cuda:0"); print("  ", tag, flush=True)
    for d in DOMAINS:
        dd, i = D[d], te[d]; st = ck["norm_stats_per_domain"][d]; y = make_Y(dd["delta"][i], (1,))[:, 0]
        vit, vm, lab, lm, S = dd["vit"][i], dd["vmask"][i], dd["lab"][i], dd["lmask"][i], dd["static"][i]
        base = roc_auc_score(y, score(enc, hd, vit, vm, lab, lm, S, st, "cuda:0")[0]); row = []
        for c in range(len(NAMES)):
            perm = rng.permutation(len(i)); v2, m2, l2, lm2 = vit.copy(), vm.copy(), lab.copy(), lm.copy()
            if c < len(VC): v2[:, :, c] = vit[perm][:, :, c]; m2[:, :, c] = vm[perm][:, :, c]
            else: k = c - len(VC); l2[:, :, k] = lab[perm][:, :, k]; lm2[:, :, k] = lm[perm][:, :, k]
            a = roc_auc_score(y, score(enc, hd, v2, m2, l2, lm2, S, st, "cuda:0")[0]); row.append((base - a) / (base - 0.5))
        IMP[d].append(row); print(f"  {tag} {d} base {base:.4f}", flush=True)
np.savez(REPO / "results" / "perm_importance_15.npz", names=np.array(NAMES), **{d: np.array(IMP[d]) for d in DOMAINS})
print("saved perm_importance_15.npz")
