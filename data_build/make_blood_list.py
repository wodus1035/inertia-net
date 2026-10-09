"""List MIMIC episodes with a positive blood culture.

Applies the mimic-code positivity rule (org_name present and not NEGATIVE/CANCELLED)
to the raw microbiologyevents and keeps blood-culture specimens only. Built over all
of MIMIC_severity, so it is independent of the cohort definition.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.common import CACHE, DATA, MIMIC_RAW
from src.config import MIMIC_SRC

OUT = f"{CACHE}/mimic_blood_positive.parquet"

sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "hadm_id"]).drop_duplicates("Episode_ID")
sev["Episode_ID"] = sev.Episode_ID.astype(str)
sev = sev.dropna(subset=["hadm_id"])
sev["h"] = sev.hadm_id.astype("int64")
hs = set(sev.h)

pos = set()
for ch in pd.read_csv(f"{MIMIC_RAW}/hosp/microbiologyevents.csv",
                      usecols=["hadm_id", "spec_type_desc", "org_name", "org_itemid"],
                      dtype={"hadm_id": "float64", "spec_type_desc": "string",
                             "org_name": "string", "org_itemid": "float64"},
                      chunksize=2_000_000):
    ch = ch.dropna(subset=["hadm_id"])
    ch["h"] = ch.hadm_id.astype("int64")
    ch = ch[ch.h.isin(hs)]
    if not len(ch):
        continue
    blood = ch.spec_type_desc.str.upper().str.contains("BLOOD CULTURE", na=False)
    ok = (ch.org_name.notna() & ~ch.org_itemid.isin([90856, 90760])
          & ch.org_name.ne("") & ch.org_name.ne("CANCELLED"))
    pos |= set(ch.loc[blood & ok, "h"])

out = sev[sev.h.isin(pos)][["Episode_ID"]]
out.to_parquet(OUT, index=False)
print(f"blood-culture-positive episodes {len(out):,}  (admissions {len(pos):,})  -> {OUT}")

coh = f"{DATA}/mimic_meta_d3_b36x2h.parquet"
if os.path.exists(coh):
    c = set(pd.read_parquet(coh, columns=["Episode_ID"]).Episode_ID.astype(str))
    n = len(set(out.Episode_ID) & c)
    print(f"  overlap with current cohort {n:,} / {len(c):,} ({n/len(c):.1%})")
