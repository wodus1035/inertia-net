"""How tightly the IV->PO switch is coupled to a care-level transition in each cohort:
hospital discharge for KNU, ICU discharge (and hospital discharge) for MIMIC.
Prints to stdout only.
"""
import numpy as np
import pandas as pd

from common import DATA, MIMIC_RAW, rule
from src.config import KNU_SRC, MIMIC_SRC


def ymd(s):
    return pd.to_datetime(s.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")


def load(dom):
    m = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet",
                        columns=["Episode_ID", "switch_day"])
    m["Episode_ID"] = m.Episode_ID.astype(str)
    return m.groupby("Episode_ID").switch_day.first().pipe(pd.to_datetime)


rule("what the switch is coupled to, per cohort")

# KNU: hospital discharge
sk = load("knu")
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Discharge_Date", "Admission_Date"])
k["Episode_ID"] = k.Episode_ID.astype(str)
k = k[k.Episode_ID.isin(set(sk.index))].drop_duplicates("Episode_ID").set_index("Episode_ID")
dk = (sk - ymd(k.Discharge_Date).reindex(sk.index)).dt.days.dropna()
print(f"\n  KNU  switch - hospital discharge (days): median {dk.median():+.0f}  "
      f"IQR {dk.quantile(.25):+.0f}~{dk.quantile(.75):+.0f}   n={len(dk):,}")
print(f"       within 1d of discharge  {(dk.abs() <= 1).mean():>6.1%}")
print(f"       2+ days before discharge {(dk <= -2).mean():>6.1%}")

# MIMIC: both ICU discharge and hospital discharge
sm = load("mimic")
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "stay_id", "Discharge_Date"]) \
        .drop_duplicates("Episode_ID")
sev["Episode_ID"] = sev.Episode_ID.astype(str)
sev = sev[sev.Episode_ID.isin(set(sm.index))].set_index("Episode_ID")
icu = pd.read_csv(f"{MIMIC_RAW}/icu/icustays.csv", usecols=["stay_id", "intime", "outtime"])
icu["out"] = pd.to_datetime(icu.outtime).dt.normalize()
icu["in_"] = pd.to_datetime(icu.intime).dt.normalize()
out = icu.set_index("stay_id").out.reindex(sev.stay_id).to_numpy()
out = pd.Series(out, index=sev.index)
dm_icu = (sm - out.reindex(sm.index)).dt.days.dropna()
dm_hosp = (sm - ymd(sev.Discharge_Date).reindex(sm.index)).dt.days.dropna()
stay = (icu.set_index("stay_id").out - icu.set_index("stay_id").in_).dt.days
print(f"\n  MIMIC switch - ICU discharge (days): median {dm_icu.median():+.0f}  "
      f"IQR {dm_icu.quantile(.25):+.0f}~{dm_icu.quantile(.75):+.0f}   n={len(dm_icu):,}")
print(f"       within 1d of ICU discharge  {(dm_icu.abs() <= 1).mean():>6.1%}")
print(f"       2+ days after ICU discharge {(dm_icu >= 2).mean():>6.1%}")
print(f"       median ICU length of stay   {stay.reindex(sev.stay_id).median():.0f}d")
print(f"\n  MIMIC switch - hospital discharge (days): median {dm_hosp.median():+.0f}  "
      f"IQR {dm_hosp.quantile(.25):+.0f}~{dm_hosp.quantile(.75):+.0f}")
print(f"       within 1d of hospital discharge {(dm_hosp.abs() <= 1).mean():>6.1%}")

rule("side by side: coupling to each cohort's own care-level transition")
print(f"  {'cohort':<10s} {'event':<20s} {'within 1d':>10s}")
print(f"  {'KNU':<10s} {'hosp discharge':<20s} {(dk.abs() <= 1).mean():>10.1%}")
print(f"  {'MIMIC':<10s} {'ICU discharge':<20s} {(dm_icu.abs() <= 1).mean():>10.1%}")
print(f"  {'MIMIC':<10s} {'hosp discharge':<20s} {(dm_hosp.abs() <= 1).mean():>10.1%}")
print("\n  The manuscript contrasts KNU 62.6% with MIMIC 21.0%, both keyed to hospital")
print("  discharge. The MIMIC observation unit is the ICU stay, and against ICU discharge")
print("  the two cohorts are coupled to a similar degree.")
