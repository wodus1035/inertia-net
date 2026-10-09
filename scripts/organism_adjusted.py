"""Organism-adjusted and organism-stratified sensitivity analysis: cohort log-OR at the
pooled median severity for MIMIC blood-culture-positive vs all KNU, before and after
adjusting for organism group, plus per-group strata. Prints to stdout only.
"""
import sys; sys.path.insert(0, "../figures")
import numpy as np, pandas as pd, statsmodels.api as sm
from common import CACHE, REPO, MIMIC_RAW, rule
from src.config import MIMIC_SRC, KNU_SRC
import paper_figures as pf
ns = {}; exec(open("make_site_xlsx.py").read().split("# cohorts")[0], ns); cls, label = ns["cls"], ns["label"]
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Organism"]).dropna(subset=["Organism"]); k["ep"] = k.Episode_ID.astype(str)
ko = k.assign(o=k.Organism.astype(str).str.split(r"\s*\|\s*")).explode("o"); ko["o"] = ko.o.str.strip(); ko = ko[ko.o.ne("")]; ko["cl"] = ko.o.map(cls)
gk = ko.groupby("ep").cl.apply(set).map(label)
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "hadm_id"]).drop_duplicates("Episode_ID"); sev["ep"] = sev.Episode_ID.astype(str)
sev = sev.dropna(subset=["hadm_id"]); sev["h"] = sev.hadm_id.astype("int64"); hs = set(sev.h); parts = []
for ch in pd.read_csv(f"{MIMIC_RAW}/hosp/microbiologyevents.csv", usecols=["hadm_id", "spec_type_desc", "org_name", "org_itemid"],
                      dtype={"hadm_id": "float64", "spec_type_desc": "string", "org_name": "string", "org_itemid": "float64"}, chunksize=2_000_000):
    ch = ch.dropna(subset=["hadm_id"]); ch["h"] = ch.hadm_id.astype("int64"); ch = ch[ch.h.isin(hs)]
    ch = ch[ch.spec_type_desc.fillna("").str.upper().str.contains("BLOOD CULTURE") & ch.org_name.notna()
            & ~ch.org_itemid.isin([90856, 90760]) & ch.org_name.ne("") & ch.org_name.ne("CANCELLED")]
    parts.append(ch[["h", "org_name"]])
mb = pd.concat(parts); mb["cl"] = mb.org_name.map(cls); gm = sev.set_index("ep").h.map(mb.groupby("h").cl.apply(set).map(label))
BL = set(pd.read_parquet(f"{CACHE}/mimic_blood_positive.parquet").Episode_ID.astype(str))
C = pf.load_cache(); s0 = np.median(np.r_[C["knu"]["s"], C["mimic"]["s"]])
def build(d, keep=None):
    eps = C[d]["eps"].astype(str); m_ = np.ones(len(eps), bool) if keep is None else np.isin(eps, list(keep))
    g = (gk if d == "knu" else gm).reindex(eps[m_]).fillna("Unknown").to_numpy()
    return pd.DataFrame({"s": C[d]["s"][m_], "y1": C[d]["Y"][m_, 0], "y2": C[d]["Y"][m_, 1], "y3": C[d]["Y"][m_, 2],
                         "pat": C[d]["pat"][m_], "grp": g, "d": 1.0 if d == "mimic" else 0.0})
K = build("knu"); M = build("mimic", BL); D = pd.concat([K, M]); D["sc"] = D.s - s0
def fit(D, adjust, h):
    X = pd.DataFrame({"const": 1.0, "s": D.sc, "d": D.d, "sd": D.sc * D.d})
    if adjust: X = pd.concat([X, pd.get_dummies(D.grp, prefix="g", drop_first=True).astype(float)], axis=1)
    r = sm.GLM(D[f"y{h}"].to_numpy().astype(float), X.to_numpy(), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": D.pat.to_numpy()})
    return r.params[2], r.bse[2]
def c50(s, y):
    r = sm.Logit(y, sm.add_constant(s)).fit(disp=0); return -r.params[0] / r.params[1]
rule("cohort log-OR at pooled median, MIMIC blood-culture-positive vs all KNU (test split, patient cluster-robust)")
print(f"  KNU anchors {len(K):,}  MIMIC blood+ anchors {len(M):,}  MIMIC label dist {M.grp.value_counts().to_dict()}")
for nm, adj in (("unadjusted", False), ("organism-group adjusted", True)):
    v = [fit(D, adj, h) for h in (1, 2, 3)]; print(f"  {nm:26s} " + "  ".join(f"h{h} {b:.2f} ({b-1.96*se:.2f}–{b+1.96*se:.2f})" for h, (b, se) in zip((1, 2, 3), v)))
rule("organism-group strata (MIMIC anchors >= 30): log-OR and c50 gap")
for grp in sorted(D.grp.unique()):
    E = D[D.grp == grp]; nk, nm_ = int((E.d == 0).sum()), int((E.d == 1).sum())
    if nm_ < 30 or nk < 30: continue
    v = [fit(E, False, h) for h in (1, 2, 3)]
    gaps = [c50(E.s[E.d == 0].to_numpy(), E[f"y{h}"][E.d == 0].to_numpy()) - c50(E.s[E.d == 1].to_numpy(), E[f"y{h}"][E.d == 1].to_numpy()) for h in (1, 2, 3)]
    print(f"  {grp:16s} KNU {nk:6,} / MIMIC {nm_:4,} anchors  logOR " + "  ".join(f"{b:.2f} ({b-1.96*se:.2f}–{b+1.96*se:.2f})" for b, se in v) + f"  gap {gaps[0]:.2f}/{gaps[1]:.2f}/{gaps[2]:.2f}")
