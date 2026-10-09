"""Cohort selection flow: counts episodes and patients surviving each filter from raw, and
compares the result with the final tensor cohort. Writes the flow to stdout.
"""
import pandas as pd
from pathlib import Path
import common  # path bootstrap
from src.config import KNU_SRC, MIMIC_SRC, DATAS, MAX_EPISODE_SPAN_DAYS, WINDOW_TAG

def d8(s):  # YYYYMMDD int -> datetime
    return pd.to_datetime(s.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")

print("="*78); print("KNU"); print("="*78)
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID","Patient_ID","Drug_Codes","Event_Date",
                                      "Admission_Date","Discharge_Date"])
k["ev"] = d8(k["Event_Date"])
code = k["Drug_Codes"].fillna("")
k["is_iv"] = code.str.startswith("DI"); k["is_po"] = code.str.startswith("DO")

g = k.groupby("Episode_ID")
ep = pd.DataFrame({
    "patient": g["Patient_ID"].first(),
    "first_rec": g["ev"].min(), "last_rec": g["ev"].max(),
    "n_iv": g["is_iv"].sum(), "n_po": g["is_po"].sum(),
})
ep["last_iv"]  = k[k.is_iv].groupby("Episode_ID")["ev"].max()
ep["last_po"]  = k[k.is_po].groupby("Episode_ID")["ev"].max()
ep["span"] = (ep["last_rec"] - ep["first_rec"]).dt.days + 1

steps = [("raw episodes in knudata.parquet", ep.index)]
s1 = ep[ep.n_iv > 0];                                    steps.append(("  + >=1 IV (DI*) day", s1.index))
s2 = s1[s1.n_po > 0];                                    steps.append(("  + >=1 oral (DO*) day", s2.index))
s3 = s2[s2.last_po > s2.last_iv];                        steps.append(("  + oral AFTER last IV (observed switch)", s3.index))
s4 = s3[s3.span <= MAX_EPISODE_SPAN_DAYS];               steps.append((f"  + span <= {MAX_EPISODE_SPAN_DAYS}d (MAX_EPISODE_SPAN_DAYS)", s4.index))

meta_k = pd.read_parquet(Path(DATAS)/f"knu_meta_{WINDOW_TAG}.parquet")
final_k = pd.Index(meta_k["Episode_ID"].unique())
steps.append(("FINAL tensor cohort (knu_meta)", final_k))

prev = None
for name, idx in steps:
    npat = ep.reindex(idx)["patient"].nunique() if name != "FINAL tensor cohort (knu_meta)" else ep.reindex(idx)["patient"].nunique()
    delta = "" if prev is None else f"   (-{len(prev)-len(idx):,})"
    print(f"{name:52s} {len(idx):>7,} ep / {npat:>6,} pt{delta}")
    prev = idx
print(f"\n  in final cohort but not in step 4: {len(final_k.difference(s4.index)):,}")
print(f"  in step 4 but not in final cohort : {len(s4.index.difference(final_k)):,}")

print(); print("="*78); print("MIMIC-IV"); print("="*78)
m = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID","Patient_ID","switch_flag","IV_Start_Date",
                                        "IV_Stop_Date","PO_Start_Date","Admission_Date","Discharge_Date"])
gm = m.groupby("Episode_ID")
epm = gm.first()
epm["span"] = (pd.to_datetime(epm["Discharge_Date"]) - pd.to_datetime(epm["Admission_Date"])).dt.days + 1

stepsm = [("raw episodes in MIMIC_severity.parquet", epm.index)]
t1 = epm[epm["IV_Start_Date"].notna() & epm["IV_Stop_Date"].notna()]; stepsm.append(("  + IV_Start & IV_Stop present", t1.index))
t2 = t1[t1["PO_Start_Date"].notna()];                                 stepsm.append(("  + PO_Start present (observed switch)", t2.index))
t3 = t2[t2["switch_flag"] == 1] if "switch_flag" in t2 else t2;       stepsm.append(("  + switch_flag == 1", t3.index))
t4 = t3[t3["span"] <= MAX_EPISODE_SPAN_DAYS];                         stepsm.append((f"  + span <= {MAX_EPISODE_SPAN_DAYS}d", t4.index))
meta_m = pd.read_parquet(Path(DATAS)/f"mimic_meta_{WINDOW_TAG}.parquet")
final_m = pd.Index(meta_m["Episode_ID"].unique())
stepsm.append(("FINAL tensor cohort (mimic_meta)", final_m))

prev = None
for name, idx in stepsm:
    npat = epm.reindex(idx)["Patient_ID"].nunique()
    delta = "" if prev is None else f"   (-{len(prev)-len(idx):,})"
    print(f"{name:52s} {len(idx):>7,} ep / {npat:>6,} pt{delta}")
    prev = idx
print(f"\n  in final cohort but not in step 4: {len(final_m.difference(t4.index)):,}")
print(f"  in step 4 but not in final cohort : {len(t4.index.difference(final_m)):,}")
