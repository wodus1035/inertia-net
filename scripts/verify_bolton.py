"""Rebuild the Bolton IV->PO switch label from raw MIMIC prescriptions and compare it
with the switch_day in our datasets; runs the same check on KNU. Prints to stdout only.

Bolton rule: routes {IV, PO/NG, PO, NU, ORAL} with {PO/NG, NU, ORAL} mapped to PO;
per stay_id collapse each route to one interval (min start, max stop); require both
routes, iv_stop <= po_stop, non-negative durations, iv_duration <= 7; label =
max(po_start, iv_stop + 1). Bolton used mimic-code's antibiotic concept on MIMIC-IV 2.0,
here 3.1 prescriptions are filtered by drug name, so exact agreement is not expected.
"""
import os
import numpy as np, pandas as pd
from common import CACHE, DATA, MIMIC_RAW
from src.config import MIMIC_SRC
# antibiotic name stems, inlined because a same-named src package is already imported
STEMS = ("penicillin ampicillin amoxicillin nafcillin oxacillin dicloxacillin piperacillin "
         "ticarcillin cefazolin cephalexin cefadroxil cefuroxime cefoxitin cefotetan cefotaxime "
         "ceftriaxone ceftazidime cefepime ceftaroline cefpodoxime cefdinir cefixime ceftolozane "
         "cefiderocol aztreonam meropenem imipenem ertapenem doripenem vaborbactam relebactam "
         "avibactam vancomycin teicoplanin daptomycin linezolid tedizolid gentamicin tobramycin "
         "amikacin streptomycin plazomicin ciprofloxacin levofloxacin moxifloxacin ofloxacin "
         "norfloxacin gemifloxacin delafloxacin azithromycin clarithromycin erythromycin "
         "clindamycin doxycycline minocycline tetracycline tigecycline eravacycline omadacycline "
         "trimethoprim sulfamethoxazole sulfadiazine metronidazole nitrofurantoin fosfomycin "
         "rifampin rifampicin colistin polymyxin chloramphenicol fidaxomicin dalbavancin "
         "oritavancin telavancin").split()

RX = str(MIMIC_RAW) + "/hosp/prescriptions.csv"
ICU = str(MIMIC_RAW) + "/icu/icustays.csv"
ROUTES = ["IV", "PO/NG", "PO", "NU", "ORAL"]

pat = "|".join(STEMS)
keep = []
n = 0
for ch in pd.read_csv(RX, usecols=["hadm_id", "starttime", "stoptime", "drug", "route"],
                      dtype={"hadm_id": "float64", "drug": "string", "route": "string"},
                      chunksize=2_000_000, low_memory=False):
    n += len(ch)
    ch = ch[ch.route.str.contains("|".join(ROUTES), na=False, case=False)]
    ch = ch[ch.drug.str.lower().str.contains(pat, na=False, regex=True)]
    keep.append(ch)
    print(f"  read {n:,}  antibiotic rows so far {sum(len(k) for k in keep):,}", flush=True)
rx = pd.concat(keep, ignore_index=True)
print(f"\nantibiotic rows {len(rx):,}   route distribution:\n{rx.route.value_counts().head(8).to_string()}")

# Bolton route mapping: NU counted as PO
rx["route2"] = rx.route.replace({"PO/NG": "PO", "NU": "PO", "ORAL": "PO"})
rx = rx[rx.route2.isin(["IV", "PO"])]
rx["s"] = pd.to_datetime(rx.starttime, errors="coerce")
rx["e"] = pd.to_datetime(rx.stoptime, errors="coerce")
rx = rx.dropna(subset=["hadm_id", "s"])

# attribute a prescription to the icustay it starts inside, as mimic-code antibiotic does
icu = pd.read_csv(ICU, usecols=["hadm_id", "stay_id", "intime", "outtime"])
icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
j = rx.merge(icu, on="hadm_id", how="inner")
j = j[(j.s >= j.intime) & (j.s <= j.outtime)]
print(f"rows attributed to an ICU stay {len(j):,}   stays {j.stay_id.nunique():,}")

sp = j.groupby(["stay_id", "route2"]).agg(st=("s", "min"), sp=("e", "max")).reset_index()
w = sp.pivot(index="stay_id", columns="route2", values=["st", "sp"])
w.columns = ["iv_start", "po_start", "iv_stop", "po_stop"]
w = w.dropna().reset_index()
for c in ["iv_start", "po_start", "iv_stop", "po_stop"]:
    w[c] = pd.to_datetime(w[c]).dt.normalize()
print(f"stays with both IV and PO {len(w):,}")

# Bolton filters
w = w[w.iv_stop <= w.po_stop]
w["iv_dur"] = (w.iv_stop - w.iv_start).dt.days
w["po_dur"] = (w.po_stop - w.po_start).dt.days
w = w[(w.iv_dur >= 0) & (w.po_dur >= 0) & (w.iv_dur <= 7)]
w["bolton_raw"] = np.maximum(w.po_start, w.iv_stop + pd.Timedelta(days=1))
print(f"stays passing Bolton filters {len(w):,}  (paper reports n=10,362; stored switch_flag=1 is 10,503)")

# compare with our dataset
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "stay_id", "switch_flag",
                                          "IV_Stop_Date", "PO_Start_Date"]).drop_duplicates("Episode_ID")
sev["Episode_ID"] = sev.Episode_ID.astype(str)
def ymd(s): return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
sev["stored_ivstop"] = ymd(sev.IV_Stop_Date); sev["stored_postart"] = ymd(sev.PO_Start_Date)
sev = sev.merge(w[["stay_id", "iv_stop", "po_start", "bolton_raw"]], on="stay_id", how="left")

inb = sev.bolton_raw.notna()
swf = sev.switch_flag == 1
print(f"\noverlap: stored switch_flag=1 {swf.sum():,} / raw repro {inb.sum():,} / "
      f"both {(swf & inb).sum():,}  (share of repro also stored {(swf & inb).sum()/max(inb.sum(),1):.1%})")

k = sev[swf & inb]
print(f"\nstored fields vs raw repro  (n={len(k):,})")
print(f"  IV_Stop_Date  match : {(k.stored_ivstop == k.iv_stop).mean():>6.1%}   "
      f"median diff {(k.stored_ivstop - k.iv_stop).dt.days.median():+.0f}d")
print(f"  PO_Start_Date match : {(k.stored_postart == k.po_start).mean():>6.1%}   "
      f"median diff {(k.stored_postart - k.po_start).dt.days.median():+.0f}d")

# final dataset first; retired variants are compared only if still on disk
variants = [("nocap  (final label)", f"{DATA}/mimic_meta_d3_b36x2h.parquet")]
for v in ("ds_ivstop", "ds_overlap", "ds_strict"):
    q = f"{CACHE}/{v}/mimic_meta_d3_b36x2h.parquet"
    if os.path.exists(q):
        variants.append((v.replace("ds_", ""), q))

for tag, path in variants:
    d = pd.read_parquet(path, columns=["Episode_ID", "switch_day"]).drop_duplicates("Episode_ID")
    d["Episode_ID"] = d.Episode_ID.astype(str); d["sw"] = pd.to_datetime(d.switch_day)
    m = k.merge(d, on="Episode_ID", how="inner").dropna(subset=["sw"])
    df = (m.sw - m.bolton_raw).dt.days
    print(f"\n[{tag}]  n={len(m):,}   vs raw-reproduced Bolton label")
    print(f"   exact {(df == 0).mean():>6.1%}   within 1d {(df.abs() <= 1).mean():>6.1%}   "
          f"median diff {df.median():+.0f}d   mean {df.mean():+.2f}")


# KNU: same Bolton rule
print("\n" + "=" * 78)
print("KNU - Bolton rule comparison")
print("=" * 78)
from src.config import KNU_SRC
kn = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Drug_Codes", "ICU"])
kn = kn.dropna(subset=["Episode_ID", "Event_Date"])
kn["Episode_ID"] = kn.Episode_ID.astype(str)
kn["d"] = pd.to_datetime(kn.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
kn = kn.dropna(subset=["d"])

# route lives in the Drug_Codes prefix; the field is a pipe-joined list, so test each code
def has(codes, pre):
    if not isinstance(codes, str): return False
    return any(c.strip().upper().startswith(pre) for c in codes.split("|") if c.strip())
kn["iv"] = kn.Drug_Codes.map(lambda c: has(c, "DI"))
kn["po"] = kn.Drug_Codes.map(lambda c: has(c, "DO"))
g = kn.groupby(["Episode_ID", "d"]).agg(iv=("iv", "any"), po=("po", "any")).reset_index()
print(f"KNU episodes {g.Episode_ID.nunique():,}  episode-days {len(g):,}")

sp = g[g.iv].groupby("Episode_ID").d.agg(iv_start="min", iv_stop="max")
pp = g[g.po].groupby("Episode_ID").d.agg(po_start="min", po_stop="max")
kw = sp.join(pp, how="inner").reset_index()
print(f"episodes with both IV and oral {len(kw):,}")
kw["iv_dur"] = (kw.iv_stop - kw.iv_start).dt.days
kw["po_dur"] = (kw.po_stop - kw.po_start).dt.days
kf = kw[(kw.iv_stop <= kw.po_stop) & (kw.iv_dur >= 0) & (kw.po_dur >= 0)]
print(f"  passing iv_stop <= po_stop {len(kf):,}")
print(f"  + passing iv_duration <= 7 {(kf.iv_dur <= 7).sum():,}   "
      f"(median IV duration {kw.iv_dur.median():.0f}d, share <=7 {(kw.iv_dur <= 7).mean():.1%})")
kf = kf.copy()
kf["bolton_raw"] = np.maximum(kf.po_start, kf.iv_stop + pd.Timedelta(days=1))

kd = pd.read_parquet(f"{DATA}/knu_meta_d3_b36x2h.parquet",
                     columns=["Episode_ID", "switch_day"]).drop_duplicates("Episode_ID")
kd["Episode_ID"] = kd.Episode_ID.astype(str); kd["sw"] = pd.to_datetime(kd.switch_day)
mm = kd.merge(kf, on="Episode_ID", how="left")
print(f"\nof our {len(kd):,} KNU episodes, {mm.bolton_raw.notna().sum():,} also pass the Bolton filters "
      f"({mm.bolton_raw.notna().mean():.1%})")
mj = mm.dropna(subset=["bolton_raw"])
dfk = (mj.sw - mj.bolton_raw).dt.days
print(f"  [nocap] exact match with Bolton label {(dfk == 0).mean():>6.1%}  within 1d {(dfk.abs() <= 1).mean():>6.1%}  "
      f"median diff {dfk.median():+.0f}d")
mj2 = mm.dropna(subset=["iv_stop"])
print(f"  our switch_day == (last IV day + 1) : "
      f"{((mj2.sw - mj2.iv_stop).dt.days == 1).mean():>6.1%}")
# was an IV-duration cap applied?
inc = mm.dropna(subset=["iv_dur"])
print(f"\n  IV duration in our cohort: median {inc.iv_dur.median():.0f}d  "
      f"<=7 {(inc.iv_dur <= 7).mean():.1%}  >7 {(inc.iv_dur > 7).mean():.1%}  max {inc.iv_dur.max():.0f}d")
