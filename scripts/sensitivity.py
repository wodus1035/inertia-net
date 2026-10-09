"""Sensitivity analysis of the site gap on the held-out test set.

Recomputes two estimands under each restriction: gap = c50(KNU) - c50(MIMIC), and the
domain odds ratio from y ~ s + domain. Also prints the Supplementary Table S8 panels.
"""
import numpy as np, pandas as pd, torch, statsmodels.api as sm
from common import CACHE, CV_TAGS, DATA, REPO, ckpt_path, load_ckpt
from src.data import load_domain, masks_of, make_Y, apply_stats
from src.triplets import build_triplets, normalize_triplet_values
from src.config import MIMIC_SRC, KNU_SRC
dev = torch.device("cuda:0")

def ymd(s): return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

D, S4 = {}, {}
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
            from src.config import patient_of
            D[dom] = dict(idx=idx, eps=d["eps"][idx], Y=make_Y(d["delta"][idx], (1,2,3)),
                          delta=d["delta"][idx], age=d["static"][idx][:,0],
                          male=d["static"][idx][:,1], anchor=None,
                          pat=patient_of(pd.Series(d["eps"][idx])).to_numpy())   # bootstrap unit = patient
    print(f"  seed {sd}", flush=True)
for dom in D: D[dom]["s"] = np.mean(S4[dom], 0)

# Anchor position within the episode; used to correct the anchor-count imbalance between cohorts
for dom in D:
    df = pd.DataFrame({"ep": D[dom]["eps"], "delta": D[dom]["delta"]})
    D[dom]["pos"] = df.groupby("ep").delta.rank(ascending=False, method="first").to_numpy()
    D[dom]["nanch"] = df.groupby("ep").ep.transform("size").to_numpy()

# ICU flag on the anchor day
mm = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "charttime", "icu_flag"])
mm["Episode_ID"] = mm.Episode_ID.astype(str); mm["d"] = pd.to_datetime(mm.charttime).dt.normalize()
mi = mm.groupby(["Episode_ID", "d"]).icu_flag.max()
kk = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Ward", "ICU"])
kk["Episode_ID"] = kk.Episode_ID.astype(str); kk["d"] = ymd(kk.Event_Date)
ki = kk.assign(icu=kk.ICU.notna()).groupby(["Episode_ID","d"]).icu.max()
for dom, tbl in [("mimic", mi), ("knu", ki)]:
    d = load_domain(dom); idx = D[dom]["idx"]
    meta = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet", columns=["Episode_ID","anchor"])
    meta["Episode_ID"] = meta.Episode_ID.astype(str)
    a = pd.to_datetime(meta.anchor).to_numpy()[idx]
    key = pd.MultiIndex.from_arrays([D[dom]["eps"], pd.to_datetime(a).normalize()])
    D[dom]["icu"] = tbl.reindex(key).fillna(0).to_numpy().astype(bool)
    print(f"  [{dom}] ICU anchor fraction {D[dom]['icu'].mean():.1%}")

# Total IV course length; the meta tables carry no course column, so rebuild it from raw
mv = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID","IV_Start_Date","IV_Stop_Date"]).drop_duplicates("Episode_ID")
mv["Episode_ID"]=mv.Episode_ID.astype(str)
mv["ivlen"]=(ymd(mv.IV_Stop_Date)-ymd(mv.IV_Start_Date)).dt.days
kd = pd.read_parquet(KNU_SRC, columns=["Episode_ID","Event_Date","Drug_Codes"]).dropna(subset=["Episode_ID","Event_Date"])
kd["Episode_ID"]=kd.Episode_ID.astype(str); kd["d"]=ymd(kd.Event_Date)
def has(c,p): return isinstance(c,str) and any(x.strip().upper().startswith(p) for x in c.split("|") if x.strip())
kiv = kd[kd.Drug_Codes.map(lambda c: has(c,"DI"))].groupby("Episode_ID").d.agg(["min","max"])
kv = (kiv["max"]-kiv["min"]).dt.days.rename("ivlen").reset_index()
for dom, tbl in [("mimic", mv[["Episode_ID","ivlen"]]), ("knu", kv)]:
    m = pd.Series(D[dom]["eps"]).map(tbl.set_index("Episode_ID").ivlen)
    D[dom]["ivlen"] = m.to_numpy()
    print(f"  [{dom}] IV course length median {np.nanmedian(D[dom]['ivlen']):.0f} d  missing {np.isnan(D[dom]['ivlen']).mean():.1%}")

def c50_of(s, y):
    if y.sum() < 5 or (1-y).sum() < 5: return np.nan
    try: r = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    except Exception: return np.nan
    return -r.params[0]/r.params[1] if r.params[1] > 0.15 else np.nan

def est(mk, mm_, hi=0, nboot=200, seed=0):
    """gap and OR with patient-cluster bootstrap CIs."""
    sk, yk, ek = D["knu"]["s"][mk], D["knu"]["Y"][mk, hi], D["knu"]["pat"][mk]
    sm_, ym, em = D["mimic"]["s"][mm_], D["mimic"]["Y"][mm_, hi], D["mimic"]["pat"][mm_]
    if min(yk.sum(), ym.sum(), (1-yk).sum(), (1-ym).sum()) < 5: return None
    gap = c50_of(sk, yk) - c50_of(sm_, ym)
    y = np.r_[yk, ym]; s_ = np.r_[sk, sm_]; dm = np.r_[np.zeros(len(yk)), np.ones(len(ym))]
    X = sm.add_constant(np.column_stack([s_, dm]))
    try: r = sm.Logit(y, X).fit(disp=0); orv = float(np.exp(r.params[2]))
    except Exception: orv = np.nan
    rng = np.random.RandomState(seed); gs, os_ = [], []
    uk, um = np.unique(ek), np.unique(em)
    ik = {u: np.where(ek==u)[0] for u in uk}; im = {u: np.where(em==u)[0] for u in um}
    for _ in range(nboot):
        a = np.concatenate([ik[u] for u in rng.choice(uk, len(uk), True)])
        b = np.concatenate([im[u] for u in rng.choice(um, len(um), True)])
        g = c50_of(sk[a], yk[a]) - c50_of(sm_[b], ym[b])
        if np.isfinite(g): gs.append(g)
        yy = np.r_[yk[a], ym[b]]; ss = np.r_[sk[a], sm_[b]]
        dd = np.r_[np.zeros(len(a)), np.ones(len(b))]
        try:
            rr = sm.Logit(yy, sm.add_constant(np.column_stack([ss, dd]))).fit(disp=0)
            os_.append(np.exp(rr.params[2]))
        except Exception: pass
    q = lambda v: (np.percentile(v,2.5), np.percentile(v,97.5)) if len(v)>=30 else (np.nan,np.nan)
    return dict(gap=gap, gap_ci=q(gs), OR=orv, or_ci=q(os_),
                nk=int(mk.sum()), nm=int(mm_.sum()),
                ek=len(np.unique(ek)), em=len(np.unique(em)))

BL = set(pd.read_parquet(f"{CACHE}/mimic_blood_positive.parquet").Episode_ID.astype(str))
K, M = D["knu"], D["mimic"]
ALLK, ALLM = np.ones(len(K["s"]), bool), np.ones(len(M["s"]), bool)
SPECS = [
 ("main analysis (no restriction)",  ALLK, ALLM),
 ("① first 7 anchors only",        K["pos"]<=7, M["pos"]<=7),
 ("② one anchor per episode",      None, None),
 ("③ ICU anchors only",            K["icu"], M["icu"]),
 ("④ non-ICU anchors only",        ~K["icu"], ~M["icu"]),
 ("⑤ IV course <= 7 d",            K["ivlen"]<=7, M["ivlen"]<=7),
 ("⑥ IV course 8-14 d",            (K["ivlen"]>=8)&(K["ivlen"]<=14), (M["ivlen"]>=8)&(M["ivlen"]<=14)),
 ("⑦ MIMIC blood-culture positive", ALLK, np.isin(M["eps"], list(BL))),
 ("⑧ age >= 65",                   K["age"]>=65, M["age"]>=65),
 ("⑨ age < 65",                    K["age"]<65,  M["age"]<65),
]
rng0 = np.random.RandomState(7)
def one_per_ep(dom):
    df = pd.DataFrame({"ep": D[dom]["eps"]}); m = np.zeros(len(df), bool)
    for _, g in df.groupby("ep").groups.items(): m[rng0.choice(np.asarray(g))] = True
    return m

print(f"\n{'='*112}\nSensitivity analysis - held-out test, h1 (switch within 1 day). 200 patient-cluster bootstraps\n{'='*112}")
print(f"  {'restriction':>26s} {'KNU anc/ep':>13s} {'MIM anc/ep':>13s} {'gap':>19s} {'domain OR':>20s}")
for nm, a, b in SPECS:
    if nm.startswith("②"): a, b = one_per_ep("knu"), one_per_ep("mimic")
    r = est(a, b)
    if r is None: print(f"  {nm:>26s}  too few samples"); continue
    g1, g2 = r["gap_ci"]; o1, o2 = r["or_ci"]
    print(f"  {nm:>26s} {r['nk']:>6,}/{r['ek']:>5,} {r['nm']:>6,}/{r['em']:>5,} "
          f"{r['gap']:>6.3f} [{g1:>6.3f},{g2:>6.3f}] {r['OR']:>7.2f} [{o1:>5.2f},{o2:>5.2f}]")


# Supplementary Table S8 panels, all three horizons from this one script
SEP = set(pd.read_csv(f"{REPO}/data/sepsis3_repro.csv")
          .query("sepsis3_repro").Episode_ID.astype(str))
m_sep = np.isin(M["eps"].astype(str), list(SEP))

m_bl = np.isin(M["eps"], list(BL))
PANELS = [("a. Sepsis-3 (MIMIC restricted)",
           [("Full cohort", ALLK, ALLM), ("Sepsis-3 subset", ALLK, m_sep)]),
          ("b. ICU (both cohorts restricted)",
           [("Full cohort", ALLK, ALLM), ("Non-ICU subset", ~K["icu"], ~M["icu"]),
            ("ICU subset", K["icu"], M["icu"])]),
          ("c. Bacteraemia (MIMIC restricted to blood-culture positive; KNU all bacteraemic by design)",
           [("Full cohort", ALLK, ALLM), ("Bacteraemia-matched", ALLK, m_bl)])]

print(f"\n{'='*112}\nSupplementary Table S8 - recomputed (c50, ds_nocap)\n{'='*112}")
for title, cols in PANELS:
    print(f"\n  {title}")
    print("    " + "horizon".ljust(9) + "".join(f"{c[0]:>30s}" for c in cols))
    for hi in range(3):
        cells = []
        for _, a, b in cols:
            r = est(a, b, hi=hi)
            if r is None:
                cells.append("too few samples".rjust(30)); continue
            g1, g2 = r["gap_ci"]
            cells.append(f"{r['gap']:>7.3f} [{g1:>6.3f},{g2:>6.3f}]".rjust(30))
        print(f"    {hi+1} day{'s' if hi else ' ':<4s}" + "".join(cells))
