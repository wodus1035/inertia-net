"""Table 1 (cohort characteristics), computed from the same tensor cohort and splits that
training and evaluation use. Writes the table rows to stdout.
"""
import numpy as np
import pandas as pd

from common import CACHE, DATA, REPO, rule
from src.config import KNU_SRC, MIMIC_SRC, patient_of
from src.data import load_domain, masks_of, make_Y

def iv_spans():
    """Total IV duration in days, rebuilt from raw. This is not the anchor-based
    'first anchor to switch' quantity: that one is deterministically iv_dur + 2."""
    out = {}
    sp = pd.read_parquet(f"{CACHE}/mimic_spans_nocap.parquet").drop_duplicates("stay_id")
    sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "stay_id"]).drop_duplicates("Episode_ID")
    sev["Episode_ID"] = sev.Episode_ID.astype(str)
    out["mimic"] = sev.merge(sp, on="stay_id").set_index("Episode_ID").iv_dur
    kn = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Drug_Codes"]).dropna()
    kn["Episode_ID"] = kn.Episode_ID.astype(str)
    has = lambda x, p: any(t.strip().upper().startswith(p) for t in str(x).split("|") if t.strip())
    kn = kn[kn.Drug_Codes.map(lambda x: has(x, "DI"))]
    kn["d"] = pd.to_datetime(kn.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    g = kn.groupby("Episode_ID").d
    out["knu"] = (g.max() - g.min()).dt.days
    return out


IV = iv_spans()
SEP = set(pd.read_csv(f"{REPO}/data/sepsis3_repro.csv").query("sepsis3_repro")
          .Episode_ID.astype(str))

# ICU on the anchor day: the only definition that is symmetric across the two sources
mm = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "charttime", "icu_flag"])
mm["Episode_ID"] = mm.Episode_ID.astype(str)
icu = {"mimic": mm.assign(d=pd.to_datetime(mm.charttime).dt.normalize())
                  .groupby(["Episode_ID", "d"]).icu_flag.max()}
kk = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Ward", "ICU"])
kk["Episode_ID"] = kk.Episode_ID.astype(str)
kk["d"] = pd.to_datetime(kk.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
icu["knu"] = kk.assign(v=kk.ICU.notna()) \
               .groupby(["Episode_ID", "d"]).v.max()

R = {}
for dom in ("knu", "mimic"):
    d = load_domain(dom)
    eps = np.asarray(d["eps"]).astype(str)
    meta = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet")
    meta["Episode_ID"] = meta.Episode_ID.astype(str)
    key = pd.MultiIndex.from_arrays([meta.Episode_ID,
                                     pd.to_datetime(meta.anchor).dt.normalize()])
    icu_anchor = icu[dom].reindex(key).fillna(0).to_numpy().astype(bool)
    pat = patient_of(pd.Series(eps)).to_numpy()
    ep1 = pd.DataFrame({"ep": eps, "pat": pat, "age": d["static"][:, 0],
                        "male": d["static"][:, 1], "delta": d["delta"],
                        "icu": icu_anchor}).groupby("ep").agg(
        pat=("pat", "first"), age=("age", "first"), male=("male", "first"),
        course=("delta", "max"), any_icu=("icu", "any"))
    te = np.where(masks_of(d, "test"))[0]
    R[dom] = dict(
        n_ep=len(ep1), n_pat=ep1.pat.nunique(),
        age=(ep1.age.mean(), ep1.age.std()),
        male=(int(ep1.male.sum()), ep1.male.mean()),
        sep=(len(set(ep1.index) & SEP), len(set(ep1.index) & SEP) / len(ep1)),
        icu_frac=icu_anchor.mean(),
        icu_pat=ep1.loc[ep1.any_icu, "pat"].nunique(),
        course=(ep1.course.median() + 1, ep1.course.quantile(.25) + 1,
                ep1.course.quantile(.75) + 1),
        ivdur=(lambda v: (v.median(), v.quantile(.25), v.quantile(.75)))(
            IV[dom].reindex(ep1.index).dropna()),
        sw=[make_Y(d["delta"][te], (h,))[:, 0].mean() for h in (1, 2, 3)],
        split={k: (len(set(eps[np.where(masks_of(d, k))[0]])),
                   int(masks_of(d, k).sum())) for k in ("train", "val", "test")})

rule("Table 1 — Cohort characteristics (ds_nocap)")
K, M = R["knu"], R["mimic"]
f = lambda a, b: f"  {a:<46s} {b}"
print(f("Episodes, n", f"{K['n_ep']:,} & {M['n_ep']:,}"))
print(f("Unique patients, n", f"{K['n_pat']:,} & {M['n_pat']:,}"))
print(f("Age, mean ± SD, years",
        f"{K['age'][0]:.1f} ± {K['age'][1]:.1f} & {M['age'][0]:.1f} ± {M['age'][1]:.1f}"))
print(f("Male sex, n (%)",
        f"{K['male'][0]:,} ({K['male'][1]:.1%}) & {M['male'][0]:,} ({M['male'][1]:.1%})"))
print(f("Met Sepsis-3 criteria, n (%)",
        f"— (100% by design) & {M['sep'][0]:,} ({M['sep'][1]:.1%})"))
print(f("ICU record on anchor day, % of anchor-days",
        f"{K['icu_frac']:.1%} & {M['icu_frac']:.1%}"))
print(f("  patients with any ICU anchor-day, n", f"{K['icu_pat']:,} & {M['icu_pat']:,}"))
print(f("IV duration (raw spans), median (IQR)",
        f"{K['ivdur'][0]:.1f} ({K['ivdur'][1]:.1f}–{K['ivdur'][2]:.1f}) & "
        f"{M['ivdur'][0]:.1f} ({M['ivdur'][1]:.1f}–{M['ivdur'][2]:.1f})"))
print(f("First anchor to switch, median (IQR)",
        f"{K['course'][0]:.1f} ({K['course'][1]:.1f}–{K['course'][2]:.1f}) & "
        f"{M['course'][0]:.1f} ({M['course'][1]:.1f}–{M['course'][2]:.1f})"))
for i, h in enumerate((1, 2, 3)):
    print(f(f"Switch within {h} day(s), % of test anchor-days",
            f"{K['sw'][i]:.1%} & {M['sw'][i]:.1%}"))
for k in ("train", "val", "test"):
    print(f(f"{k.capitalize()}, episodes (anchor-days)",
            f"{K['split'][k][0]:,} ({K['split'][k][1]:,}) & "
            f"{M['split'][k][0]:,} ({M['split'][k][1]:,})"))
