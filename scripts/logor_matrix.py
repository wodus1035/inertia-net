"""Cohort effect by condition x horizon (Supplementary logor_matrix table).

Fits y(h) ~ s + domain + s:domain (binomial GLM, patient cluster-robust SE). Because of
the interaction, logOR(s0) = b_domain + b_interaction * s0 is evaluated at the median and
90th percentile of the pooled s, with delta-method variance. domain is MIMIC = 1, so a
positive value means MIMIC switches sooner at the same readiness.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import CV_TAGS, DATA, REPO, ckpt_path, load_ckpt, rule, score_split
from src.config import MIMIC_SRC, patient_of

DOMS = ("knu", "mimic")
D = {}
for sd in CV_TAGS:
    enc, hd, ck = load_ckpt(ckpt_path(sd), "cuda:0")
    for dom in DOMS:
        r = score_split(None, dom, "test", "cuda:0", enc, hd, ck)
        D.setdefault(dom, r).setdefault("S", []).append(r["s"])
    print(f"  seed {sd}", flush=True)
for dom in DOMS:
    D[dom]["s"] = np.mean(D[dom]["S"], 0)
    df = pd.DataFrame({"ep": D[dom]["eps"], "delta": D[dom]["delta"]})
    D[dom]["pos"] = df.groupby("ep").delta.rank(ascending=False, method="first").to_numpy()

# ICU flag on the anchor day, same definition as in sensitivity.py
mm = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "charttime", "icu_flag"])
mm["Episode_ID"] = mm.Episode_ID.astype(str)
mm["d"] = pd.to_datetime(mm.charttime).dt.normalize()
icu_tbl = {"mimic": mm.groupby(["Episode_ID", "d"]).icu_flag.max()}
from src.config import KNU_SRC
kk = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Ward", "ICU"])
kk["Episode_ID"] = kk.Episode_ID.astype(str)
kk["d"] = pd.to_datetime(kk.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
icu_tbl["knu"] = kk.assign(icu=kk.ICU.notna()) \
                   .groupby(["Episode_ID", "d"]).icu.max()
from src.data import load_domain, masks_of
for dom in DOMS:
    # meta holds every anchor, so subset to the test rows for the mask lengths to match
    idx = np.where(masks_of(load_domain(dom), "test"))[0]
    m = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet",
                        columns=["Episode_ID", "anchor"]).iloc[idx]
    m["Episode_ID"] = m.Episode_ID.astype(str)
    key = pd.MultiIndex.from_arrays([m.Episode_ID, pd.to_datetime(m.anchor).dt.normalize()])
    D[dom]["icu"] = icu_tbl[dom].reindex(key).fillna(0).to_numpy().astype(bool)
    D[dom]["pat"] = patient_of(pd.Series(D[dom]["eps"])).to_numpy()

BL = set(pd.read_parquet(f"{REPO}/data/cache/mimic_blood_positive.parquet").Episode_ID.astype(str))
SEP = set(pd.read_csv(f"{REPO}/data/sepsis3_repro.csv").query("sepsis3_repro")
          .Episode_ID.astype(str))
K, M = D["knu"], D["mimic"]
ALLK, ALLM = np.ones(len(K["s"]), bool), np.ones(len(M["s"]), bool)


def pos_weights(mk, mm_):
    """Weights that match each cohort's anchor-position distribution to the pooled one."""
    pk, pm = np.clip(K["pos"][mk], 1, 20), np.clip(M["pos"][mm_], 1, 20)
    tgt = pd.Series(np.r_[pk, pm]).value_counts(normalize=True)
    out = []
    for p in (pk, pm):
        cur = pd.Series(p).value_counts(normalize=True)
        w = pd.Series(p).map(tgt / cur).to_numpy()
        out.append(w / w.mean())
    return out


def fit(mk, mm_, hi, w=None):
    """logOR(s0) and SE at the pooled median and p90, patient cluster-robust."""
    y = np.r_[K["Y"][mk, hi], M["Y"][mm_, hi]]
    s = np.r_[K["s"][mk], M["s"][mm_]]
    dm = np.r_[np.zeros(mk.sum()), np.ones(mm_.sum())]
    g = np.r_[K["pat"][mk], M["pat"][mm_]]
    X = sm.add_constant(np.column_stack([s, dm, s * dm]))
    kw = dict(freq_weights=np.r_[w[0], w[1]]) if w is not None else {}
    r = sm.GLM(y, X, family=sm.families.Binomial(), **kw).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    b, V = r.params, r.cov_params()
    out = {}
    for nm, s0 in (("median", np.median(s)), ("p90", np.percentile(s, 90))):
        est = b[2] + b[3] * s0
        var = V[2, 2] + s0 ** 2 * V[3, 3] + 2 * s0 * V[2, 3]
        se = float(np.sqrt(max(var, 0)))
        out[nm] = (float(est), est - 1.96 * se, est + 1.96 * se)
    return out


CONDS = [
    ("All anchors",                ALLK, ALLM, False),
    ("First 7 anchor-days",        K["pos"] <= 7, M["pos"] <= 7, False),
    ("Anchor-position reweighted", ALLK, ALLM, True),
    ("ICU",                        K["icu"], M["icu"], False),
    ("Non-ICU",                    ~K["icu"], ~M["icu"], False),
    ("Sepsis-3",                   ALLK, np.isin(M["eps"].astype(str), list(SEP)), False),
    ("Bacteraemia",                ALLK, np.isin(M["eps"].astype(str), list(BL)), False),
]

for point in ("median", "p90"):
    rule(f"cohort log-odds ratio, evaluated at pooled {point}   (MIMIC = 1)")
    print(f"  {'condition':<28s}" + "".join(f"{'h'+str(h+1):>26s}" for h in range(3)))
    for nm, a, b_, rw in CONDS:
        cells = []
        for hi in range(3):
            try:
                w = pos_weights(a, b_) if rw else None
                e, lo, hi_ = fit(a, b_, hi, w)[point]
                cells.append(f"{e:>7.2f} ({lo:>5.2f}-{hi_:>5.2f})".rjust(26))
            except Exception as ex:
                cells.append(f"failed {type(ex).__name__}".rjust(26))
        print(f"  {nm:<28s}" + "".join(cells))
