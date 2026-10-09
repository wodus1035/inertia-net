"""Split the KNU cohort into its two hospitals (KNUH, KNUHCH) and place both on the score
axis, to compare the within-university gap against the KNU-MIMIC gap.
"""
import numpy as np, pandas as pd, torch, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from common import CV_TAGS, ckpt_path, load_ckpt
from src.data import load_domain, masks_of, make_Y, apply_stats
from src.triplets import build_triplets, normalize_triplet_values
from src.config import KNU_SRC
dev = torch.device("cuda:0")
def ymd(s): return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

S4, D = {}, {}
for sd in CV_TAGS:
    enc, hd, ck = load_ckpt(ckpt_path(sd), dev)
    h = dict(ck["hp"])
    for dom in ["knu", "mimic"]:
        d = load_domain(dom); idx = np.where(masks_of(d, "test"))[0]
        st = ck["norm_stats_per_domain"][dom]
        trip, valid = build_triplets(d["vit"][idx], d["vmask"][idx], d["lab"][idx], d["lmask"][idx])
        vi = normalize_triplet_values(trip, np.concatenate([st["vc"], st["lc"]]),
                                      np.concatenate([st["vs"], st["ls"]]))
        Sx = apply_stats(d["static"][idx].copy()[:, None, :], st["sc"], st["ss"])[:, 0, :]
        o = []
        with torch.no_grad():
            for i in range(0, len(vi), 512):
                _, f = enc(torch.tensor(vi[i:i+512]).to(dev), torch.tensor(valid[i:i+512, :, None]).to(dev),
                           torch.tensor(Sx[i:i+512]).float().to(dev), return_fused=True)
                o.append(hd.score_head(f).cpu().numpy().ravel())
        S4.setdefault(dom, []).append(np.concatenate(o))
        if sd == CV_TAGS[0]:
            D[dom] = dict(eps=d["eps"][idx], Y=make_Y(d["delta"][idx], (1,2,3)),
                          delta=d["delta"][idx], age=d["static"][idx][:,0], idx=idx)
    print(f"  seed {sd}", flush=True)
for dom in D: D[dom]["s"] = np.mean(S4[dom], 0)

k = pd.read_parquet(KNU_SRC, columns=["Episode_ID","source_inst","Event_Date","Drug_Codes","Ward","ICU"])
k["Episode_ID"] = k.Episode_ID.astype(str)
inst = k.groupby("Episode_ID").source_inst.first()
D["knu"]["inst"] = pd.Series(D["knu"]["eps"]).map(inst).to_numpy()
k["d"] = ymd(k.Event_Date)
def has(c,p): return isinstance(c,str) and any(x.strip().upper().startswith(p) for x in c.split("|") if x.strip())
iv = k[k.Drug_Codes.map(lambda c: has(c,"DI"))].groupby("Episode_ID").d.agg(["min","max"])
ivlen = (iv["max"]-iv["min"]).dt.days.rename("ivlen")
D["knu"]["ivlen"] = pd.Series(D["knu"]["eps"]).map(ivlen).to_numpy()
icu = k.assign(i=k.ICU.notna()).groupby("Episode_ID").i.mean()
D["knu"]["icu"] = pd.Series(D["knu"]["eps"]).map(icu).to_numpy()

def c50_se(s, y, eps, nboot=300, seed=0):
    if y.sum() < 5 or (1-y).sum() < 5: return None
    r = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    b0, b1 = r.params
    if b1 <= 0.15: return None
    c = -b0/b1
    rng = np.random.RandomState(seed); ug = np.unique(eps)
    ix = {u: np.where(eps==u)[0] for u in ug}; bs = []
    for _ in range(nboot):
        m = np.concatenate([ix[u] for u in rng.choice(ug, len(ug), True)])
        if y[m].sum() < 5 or (1-y[m]).sum() < 5: continue
        try:
            rr = sm.Logit(y[m], sm.add_constant(s[m])).fit(disp=0)
            if rr.params[1] > 0.15: bs.append(-rr.params[0]/rr.params[1])
        except Exception: pass
    bs = np.array(bs)
    return dict(c50=c, se=bs.std(), ci=(np.percentile(bs,2.5), np.percentile(bs,97.5)),
                n=len(y), ep=len(ug), prev=y.mean(), auroc=roc_auc_score(y, s),
                slope=b1)

print(f"\n{'='*112}\nKNU split into its two hospitals - held-out test, h1\n{'='*112}")
print(f"  {'site':>14s} {'anchor':>8s} {'ep':>6s} {'prev':>7s} {'IV med':>7s} {'ICU%':>6s} "
      f"{'AUROC':>7s} {'c50':>7s} {'SE':>6s} {'95% CI':>16s}")
rows = {}
K = D["knu"]
for nm, m in [("KNU overall", np.ones(len(K["s"]), bool)),
              ("  KNUH", K["inst"]=="KNUH"),
              ("  KNUHCH", K["inst"]=="KNUHCH")]:
    r = c50_se(K["s"][m], K["Y"][m,0], K["eps"][m])
    rows[nm.strip()] = r
    print(f"  {nm:>14s} {r['n']:>8,} {r['ep']:>6,} {r['prev']:>7.1%} "
          f"{np.nanmedian(K['ivlen'][m]):>7.0f} {np.nanmean(K['icu'][m]):>6.1%} "
          f"{r['auroc']:>7.4f} {r['c50']:>7.3f} {r['se']:>6.3f} [{r['ci'][0]:>6.3f},{r['ci'][1]:>6.3f}]")
M = D["mimic"]
rm = c50_se(M["s"], M["Y"][:,0], M["eps"])
rows["MIMIC"] = rm
print(f"  {'MIMIC-IV':>14s} {rm['n']:>8,} {rm['ep']:>6,} {rm['prev']:>7.1%} {'3':>7s} {'69.7%':>6s} "
      f"{rm['auroc']:>7.4f} {rm['c50']:>7.3f} {rm['se']:>6.3f} [{rm['ci'][0]:>6.3f},{rm['ci'][1]:>6.3f}]")

a, b = rows["KNUH"], rows["KNUHCH"]
d_kk = a["c50"] - b["c50"]; se_kk = np.hypot(a["se"], b["se"])
d_km = rows["KNU overall"]["c50"] - rm["c50"]
print(f"\n{'='*112}\nthe two gaps compared\n{'='*112}")
print(f"  KNUH − KNUHCH        {d_kk:>+7.3f}  SE {se_kk:.3f}  95% CI [{d_kk-1.96*se_kk:+.3f}, {d_kk+1.96*se_kk:+.3f}]  z={d_kk/se_kk:.1f}")
print(f"  KNU overall - MIMIC  {d_km:>+7.3f}")
print(f"  -> within-university gap is {abs(d_kk)/abs(d_km):.0%} of the KNU-MIMIC gap")
print(f"  -> {abs(d_kk)/0.287:.2f} x the between-hospital SD tau=0.287 reported for US hospitals")
