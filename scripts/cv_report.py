"""Reporting of the 5-fold x 3-seed cross-validation.

Prints (A) across-fold mean +/- SD with 95% CI, (B) pooled out-of-fold predictions with a
patient-cluster bootstrap CI, and (C) held-out test performance and the c50 gap.
"""
import numpy as np, pandas as pd, torch, statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score
from common import MODELS, ckpt_path, load_ckpt
from src.train_inertia_net import _make_cv_fold_masks
from src.data import load_domain, masks_of, make_Y, apply_stats
from src.triplets import build_triplets, normalize_triplet_values
from src.config import patient_of
FOLDS, SEEDS = [0, 1, 2, 3, 4], [42, 43, 44]
dev = torch.device("cuda:0")

def load(f, s):
    p = ckpt_path(f"cv_f{f}_s{s}")
    enc, hd, ck = load_ckpt(p, dev)
    h = dict(ck["hp"])
    return ck, enc, hd

_FCACHE = {}
def feats(dom, idx, stats_key):
    k = (dom, len(idx), int(idx[0]), int(idx[-1]), stats_key)
    if k in _FCACHE: return _FCACHE[k]
    d = D[dom]
    trip, valid = build_triplets(d["vit"][idx], d["vmask"][idx], d["lab"][idx], d["lmask"][idx])
    _FCACHE[k] = (trip, valid, d["static"][idx].copy())
    return _FCACHE[k]

D = {dom: load_domain(dom) for dom in ["knu", "mimic"]}
TE = {dom: np.where(masks_of(D[dom], "test"))[0] for dom in D}

def score(enc, hd, dom, idx, st):
    d = D[dom]
    trip, valid = build_triplets(d["vit"][idx], d["vmask"][idx], d["lab"][idx], d["lmask"][idx])
    vi = normalize_triplet_values(trip, np.concatenate([st["vc"], st["lc"]]),
                                  np.concatenate([st["vs"], st["ls"]]))
    S = apply_stats(d["static"][idx].copy()[:, None, :], st["sc"], st["ss"])[:, 0, :]
    o = []
    with torch.no_grad():
        for i in range(0, len(vi), 512):
            _, f_ = enc(torch.tensor(vi[i:i+512]).to(dev), torch.tensor(valid[i:i+512, :, None]).to(dev),
                        torch.tensor(S[i:i+512]).float().to(dev), return_fused=True)
            o.append(hd.score_head(f_).cpu().numpy().ravel())
    return np.concatenate(o)

def c50(s, y):
    if y.sum() < 5 or (1-y).sum() < 5: return np.nan
    try: r = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    except Exception: return np.nan
    return -r.params[0]/r.params[1] if r.params[1] > 0.15 else np.nan

VAL, TEST, OOF = {}, {}, {dom: {} for dom in D}
for f in FOLDS:
    for s in SEEDS:
        try: ck, enc, hd = load(f, s)
        except Exception as e: print(f"  f{f}s{s} missing"); continue
        for dom in D:
            comb = masks_of(D[dom], "train") | masks_of(D[dom], "val")
            vi_ = np.where(_make_cv_fold_masks(comb, D[dom]["eps"], 5, 42)[f])[0]
            st = ck["norm_stats_per_domain"][dom]
            sv = score(enc, hd, dom, vi_, st); Yv = make_Y(D[dom]["delta"][vi_], (1,2,3))
            stt = score(enc, hd, dom, TE[dom], st); Yt = make_Y(D[dom]["delta"][TE[dom]], (1,2,3))
            VAL[(f,s,dom)] = dict(auroc=[roc_auc_score(Yv[:,h], sv) for h in range(3)],
                                  c50=[c50(sv, Yv[:,h]) for h in range(3)])
            TEST[(f,s,dom)] = dict(auroc=[roc_auc_score(Yt[:,h], stt) for h in range(3)],
                                   auprc=[average_precision_score(Yt[:,h], stt) for h in range(3)],
                                   c50=[c50(stt, Yt[:,h]) for h in range(3)], s=stt)
            OOF[dom].setdefault(s, {}).update({i: v for i, v in zip(vi_, sv)})
        print(f"  f{f} s{s} done", flush=True)

def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return v.mean(), v.std(ddof=1), v.mean()-1.96*v.std(ddof=1)/np.sqrt(len(v)), v.mean()+1.96*v.std(ddof=1)/np.sqrt(len(v)), len(v)

print(f"\n{'='*104}\n(A) across-fold variation - 5 folds x 3 seeds = {len(FOLDS)*len(SEEDS)} models\n{'='*104}")
for dom in D:
    print(f"\n[{dom.upper()}]")
    print(f"  {'':10s} {'val AUROC (mean±SD)':>26s} {'test AUROC (mean±SD)':>26s} {'test 95% CI':>20s}")
    for h in range(3):
        va = ms([VAL[k]["auroc"][h] for k in VAL if k[2]==dom])
        te = ms([TEST[k]["auroc"][h] for k in TEST if k[2]==dom])
        print(f"  h{h+1:<9d} {va[0]:>10.4f} ± {va[1]:.4f} (n={va[4]:>2d}) {te[0]:>10.4f} ± {te[1]:.4f} (n={te[4]:>2d}) "
              f"{te[2]:>9.4f}–{te[3]:.4f}")
print(f"\n{'='*104}\n(B) pooled out-of-fold - every train+val anchor predicted once, patient bootstrap CI\n{'='*104}")
rng = np.random.RandomState(0)
for dom in D:
    comb = np.where(masks_of(D[dom], "train") | masks_of(D[dom], "val"))[0]
    Y = make_Y(D[dom]["delta"], (1,2,3))
    pat = patient_of(pd.Series(D[dom]["eps"])).to_numpy()
    sv = np.full(len(D[dom]["eps"]), np.nan)
    for i in comb:
        vals = [OOF[dom][s][i] for s in SEEDS if i in OOF[dom].get(s, {})]
        if vals: sv[i] = np.mean(vals)
    ok = np.isfinite(sv)
    print(f"\n[{dom.upper()}]  OOF predictions {ok.sum():,} / {len(comb):,} anchors   patients {len(set(pat[ok])):,}")
    for h in range(3):
        y = Y[ok, h]; s_ = sv[ok]; g = pat[ok]
        a = roc_auc_score(y, s_)
        ug = np.unique(g); idx = {u: np.where(g==u)[0] for u in ug}
        bs = []
        for _ in range(300):
            pick = rng.choice(ug, len(ug), replace=True)
            m = np.concatenate([idx[u] for u in pick])
            if 0 < y[m].sum() < len(m): bs.append(roc_auc_score(y[m], s_[m]))
        bs = np.array(bs)
        print(f"  h{h+1}  AUROC {a:.4f}  [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]  (300 patient-cluster bootstraps)")
print(f"\n{'='*104}\n(C) gap = c50(KNU) − c50(MIMIC),  held-out test\n{'='*104}")
for h in range(3):
    g = [TEST[(f,s,'knu')]["c50"][h] - TEST[(f,s,'mimic')]["c50"][h] for f in FOLDS for s in SEEDS
         if (f,s,'knu') in TEST and (f,s,'mimic') in TEST]
    r = ms(g)
    print(f"  h{h+1}  gap {r[0]:.4f} ± {r[1]:.4f}   95% CI {r[2]:.4f}–{r[3]:.4f}   (n={r[4]})")
