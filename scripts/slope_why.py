"""Why the s->switch slope for h1 differs between cohorts: splits episodes by whether the
switch coincides with a care-level transition, and checks logistic linearity with
non-parametric log-odds by pooled-score decile. Prints to stdout only.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import CV_TAGS, DATA, MIMIC_RAW, rule, score_models
from src.config import KNU_SRC, MIMIC_SRC

S, D = score_models(CV_TAGS, ("knu", "mimic"), split="test", verbose=False)
s = {d: np.mean(S[d], 0) for d in S}; y = {d: D[d]["Y"][:, 0] for d in S}; eps = {d: np.asarray(D[d]["eps"]).astype(str) for d in S}

def slope(x, t):
    if t.sum() < 5: return np.nan, np.nan
    r = sm.Logit(t, sm.add_constant(x)).fit(disp=0); return r.params[1], r.bse[1]

def ymd(v): return pd.to_datetime(v.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
# switch day minus care-transition day, per episode
def sw(dom):
    m = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet", columns=["Episode_ID", "switch_day"]); m["Episode_ID"] = m.Episode_ID.astype(str)
    return pd.to_datetime(m.groupby("Episode_ID").switch_day.first())
swk, swm = sw("knu"), sw("mimic")
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Discharge_Date"]); k["Episode_ID"] = k.Episode_ID.astype(str)
k = k.drop_duplicates("Episode_ID").set_index("Episode_ID"); dk = (swk - ymd(k.Discharge_Date).reindex(swk.index)).dt.days
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "stay_id"]).drop_duplicates("Episode_ID"); sev["Episode_ID"] = sev.Episode_ID.astype(str); sev = sev.set_index("Episode_ID")
icu = pd.read_csv(f"{MIMIC_RAW}/icu/icustays.csv", usecols=["stay_id", "outtime"]); icu["out"] = pd.to_datetime(icu.outtime).dt.normalize()
out = pd.Series(icu.set_index("stay_id").out.reindex(sev.stay_id).to_numpy(), index=sev.index); dm = (swm - out.reindex(swm.index)).dt.days

rule("h1 slope, all anchors (s averaged over the 15 models, held-out test)")
for d in ("knu", "mimic"):
    b, se = slope(s[d], y[d]); print(f"  {d.upper():6s} b1 {b:.3f} ±{se:.3f}   n {len(y[d]):,}  positive {y[d].mean():.1%}")

rule("split by whether the switch coincides with the care-level transition (KNU discharge / MIMIC ICU discharge)")
for d, off, label in (("mimic", dm, "ICU discharge"), ("knu", dk, "hosp discharge")):
    o = off.reindex(eps[d]).to_numpy()
    for nm, msk in ((f"{label} within 1d", np.abs(o) <= 1), (f"{label} 2d+ apart", np.abs(o) >= 2)):
        msk = msk & np.isfinite(o)
        b, se = slope(s[d][msk], y[d][msk])
        print(f"  {d.upper():6s} {nm:20s} b1 {b:.3f} ±{se:.3f}   anchors {msk.sum():>6,}  positive {y[d][msk].mean():.1%}")

rule("non-parametric: log-odds by pooled-s decile (under linearity the slope is decile-invariant)")
edges = np.quantile(np.r_[s["knu"], s["mimic"]], np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
print(f"  {'dec':>3s} {'s med':>7s} {'KNU logit':>10s} {'MIMIC logit':>12s} {'diff':>6s}")
lo_k, lo_m, xs = [], [], []
for i in range(10):
    bk = (s["knu"] >= edges[i]) & (s["knu"] < edges[i+1]); bm = (s["mimic"] >= edges[i]) & (s["mimic"] < edges[i+1])
    pk, pm = y["knu"][bk].mean(), y["mimic"][bm].mean(); xm = np.median(np.r_[s["knu"][bk], s["mimic"][bm]])
    lk = np.log(pk/(1-pk)) if 0 < pk < 1 else np.nan; lm = np.log(pm/(1-pm)) if 0 < pm < 1 else np.nan
    lo_k.append(lk); lo_m.append(lm); xs.append(xm)
    print(f"  {i+1:>3d} {xm:>7.2f} {lk:>10.2f} {lm:>12.2f} {lm-lk:>6.2f}")
xs, lo_k, lo_m = map(np.array, (xs, lo_k, lo_m)); ok = np.isfinite(lo_k) & np.isfinite(lo_m)
lo_half = xs < np.median(xs)
for nm, m_ in (("lower half", ok & lo_half), ("upper half", ok & ~lo_half)):
    bk_ = np.polyfit(xs[m_], lo_k[m_], 1)[0]; bm_ = np.polyfit(xs[m_], lo_m[m_], 1)[0]
    print(f"  segment slope {nm}:  KNU {bk_:.2f}   MIMIC {bm_:.2f}")
