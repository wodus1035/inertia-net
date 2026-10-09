"""Reproduce mimic-code's Sepsis-3 flag from the raw MIMIC tables for our cohort and
compare it with the stored flags file; writes {REPO}/data/sepsis3_repro.csv.

Follows antibiotic.sql (drug-name patterns, drug_type != BASE, ophthalmic/otic/topical
routes dropped), suspicion_of_infection (culture->antibiotic 72h or antibiotic->culture
24h, regardless of positivity) and sepsis3.sql (SOFA >= 2 within [-48h, +24h] of the
suspicion time, per ICU stay). SOFA is taken from MIMIC_severity rather than mimic-code's
derived sofa table, which we do not have.
"""
import numpy as np, pandas as pd
from common import DATA, MIMIC_RAW, PAPER, REPO, RESULTS
from src.config import MIMIC_SRC
M = str(MIMIC_RAW) + ""

pats = [l.strip() for l in open(f"{RESULTS}/abx_patterns.txt") if l.strip()]
print(f"antibiotic.sql patterns: {len(pats)}")
rx_pat = "|".join(p.replace("-", r"\-") for p in pats)

# 1. antibiotic
BAD_ROUTE = {"OU", "OS", "OD", "AU", "AS", "AD", "TP"}
keep, n = [], 0
for ch in pd.read_csv(f"{M}/hosp/prescriptions.csv",
                      usecols=["subject_id", "hadm_id", "starttime", "drug", "drug_type", "route"],
                      dtype={"subject_id": "int64", "hadm_id": "float64", "drug": "string",
                             "drug_type": "string", "route": "string"},
                      chunksize=2_000_000, low_memory=False):
    n += len(ch)
    ch = ch[ch.drug_type.ne("BASE") & ch.route.notna()]
    r = ch.route.str.upper()
    ch = ch[~r.isin(BAD_ROUTE) & ~r.str.contains("EAR|EYE", na=False)]
    d = ch.drug.str.lower()
    ch = ch[d.str.contains(rx_pat, na=False, regex=True)
            & ~d.str.contains("cream|desensitization|ophth oint|gel", na=False)]
    keep.append(ch[["subject_id", "hadm_id", "starttime"]])
    print(f"  prescriptions {n:,} -> antibiotic rows so far {sum(len(k) for k in keep):,}", flush=True)
ab = pd.concat(keep, ignore_index=True)
ab["t"] = pd.to_datetime(ab.starttime, errors="coerce")
ab = ab.dropna(subset=["t"])
print(f"antibiotic prescriptions {len(ab):,}  patients {ab.subject_id.nunique():,}")

# 2. microbiology, one row per specimen (positivity is irrelevant here)
mb = pd.read_csv(f"{M}/hosp/microbiologyevents.csv",
                 usecols=["micro_specimen_id", "subject_id", "chartdate", "charttime",
                          "spec_type_desc", "org_name", "org_itemid"],
                 dtype={"subject_id": "int64", "org_name": "string", "spec_type_desc": "string"})
mb["ct"] = pd.to_datetime(mb.charttime, errors="coerce")
mb["cd"] = pd.to_datetime(mb.chartdate, errors="coerce")
mb["t"] = mb.ct.fillna(mb.cd)
me = mb.dropna(subset=["t"]).groupby("micro_specimen_id").agg(
    subject_id=("subject_id", "max"), t=("t", "max")).reset_index()
print(f"culture specimens {len(me):,}  (not filtered on positivity, as in the SQL)")

# 3. suspicion of infection: culture->antibiotic within 72h, or antibiotic->culture within 24h
# matched by searchsorted of antibiotic times into each patient's sorted culture times
me = me.sort_values(["subject_id", "t"])
ab = ab.sort_values(["subject_id", "t"])
grp_t = {sid: g.t.to_numpy("datetime64[ns]") for sid, g in me.groupby("subject_id", sort=False)}
H72 = np.timedelta64(72, "h"); H24 = np.timedelta64(24, "h")
soi_t = np.full(len(ab), np.datetime64("NaT"), dtype="datetime64[ns]")
sids = ab.subject_id.to_numpy(); ats = ab.t.to_numpy("datetime64[ns]")
start = 0
for sid, g in ab.groupby("subject_id", sort=False):
    k = len(g); sl = slice(start, start + k); start += k
    ct = grp_t.get(sid)
    if ct is None: continue
    a = ats[sl]
    # antibiotic within 72h after a culture -> soi time is that culture time
    i = np.searchsorted(ct, a, side="right") - 1
    ok = (i >= 0)
    prev = np.where(ok, ct[np.clip(i, 0, len(ct)-1)], np.datetime64("NaT"))
    hit72 = ok & (a > prev) & (a <= prev + H72)
    # culture within 24h after an antibiotic -> soi time is the antibiotic time
    j = np.searchsorted(ct, a, side="left")
    ok2 = (j < len(ct))
    nxt = np.where(ok2, ct[np.clip(j, 0, len(ct)-1)], np.datetime64("NaT"))
    hit24 = ok2 & (a >= nxt - H24) & (a < nxt)
    soi_t[sl] = np.where(hit72, prev, np.where(hit24, a, np.datetime64("NaT")))
ab["soi_t"] = soi_t
soi = ab.dropna(subset=["soi_t"]).copy()
print(f"antibiotic prescriptions meeting suspicion of infection {len(soi):,} / {len(ab):,} ({len(soi)/len(ab):.1%})")

# 4. SOFA >= 2 (from MIMIC_severity) within the [-48h, +24h] window
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "subject_id", "hadm_id", "charttime", "SOFA"])
sev["Episode_ID"] = sev.Episode_ID.astype(str)
sev["ct"] = pd.to_datetime(sev.charttime, errors="coerce")
sev = sev.dropna(subset=["ct"])
hi = sev[sev.SOFA >= 2]
print(f"MIMIC_severity rows {len(sev):,}   SOFA>=2 rows {len(hi):,} ({len(hi)/len(sev):.1%})")

coh = pd.read_parquet(f"{DATA}/mimic_meta_d3_b36x2h.parquet",
                      columns=["Episode_ID"]).Episode_ID.astype(str).unique()
epi = sev[sev.Episode_ID.isin(coh)][["Episode_ID", "subject_id"]].drop_duplicates()
print(f"cohort episodes {len(epi):,}")

# an episode is sepsis3 if any soi time of that patient falls in [-24h, +48h] of a SOFA>=2 row
hi_by = {s: g.ct.sort_values().to_numpy("datetime64[ns]") for s, g in
         hi[hi.Episode_ID.isin(coh)].groupby("Episode_ID", sort=False)}
soi_by = {s: g.soi_t.sort_values().to_numpy("datetime64[ns]") for s, g in soi.groupby("subject_id", sort=False)}
H48 = np.timedelta64(48, "h")
res = []
for ep, sid in epi.itertuples(index=False):
    st = soi_by.get(sid); sf = hi_by.get(ep)
    if st is None or sf is None or len(st) == 0 or len(sf) == 0:
        res.append((ep, False)); continue
    # sofa endtime in [soi-48h, soi+24h]  <=>  soi in [sofa-24h, sofa+48h]
    i = np.searchsorted(sf, st, side="left")
    lo = np.where(i > 0, sf[np.clip(i-1, 0, len(sf)-1)], np.datetime64("NaT"))
    hi2 = np.where(i < len(sf), sf[np.clip(i, 0, len(sf)-1)], np.datetime64("NaT"))
    ok = (((st >= lo - H24) & (st <= lo + H48)) | ((st >= hi2 - H24) & (st <= hi2 + H48)))
    res.append((ep, bool(np.nansum(ok) > 0) if ok.size else False))
out = pd.DataFrame(res, columns=["Episode_ID", "sepsis3_repro"])
print(f"\n{'='*70}\nreproduced Sepsis-3 positive {out.sepsis3_repro.sum():,} / {len(out):,} "
      f"({out.sepsis3_repro.mean():.1%})\n{'='*70}")

ref = pd.read_csv(f"{PAPER}/analysis_outputs/dat/mimic_sepsis3_flags_mimiccode.csv")
ref["Episode_ID"] = ref.Episode_ID.astype(str)
m = out.merge(ref, on="Episode_ID", how="inner")
print(f"episodes overlapping the stored flags file: {len(m):,}")
print(f"  stored 63.5% / reproduced {m.sepsis3_repro.mean():.1%}")
print(f"  agreement {(m.sepsis3_repro == m.sepsis3_flag).mean():.1%}")
print(pd.crosstab(m.sepsis3_flag, m.sepsis3_repro, rownames=["stored"], colnames=["repro"]).to_string())
out.to_csv(f"{REPO}/data/sepsis3_repro.csv", index=False)
print(f"\n-> {REPO}/data/sepsis3_repro.csv")
