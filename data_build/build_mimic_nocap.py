"""Rebuild the MIMIC cohort from the raw tables without the 7-day IV cap.

MIMIC_severity only carries IV_Start/IV_Stop/PO_Start for the Bolton cohort, which
has an iv_duration <= 7 filter baked in; KNU has no such cap, so the two cohorts are
asymmetric. This applies the Bolton rules to prescriptions.csv minus that filter and
injects the result into raw before running the usual preprocessing pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.common import CACHE, DATA, MIMIC_RAW, V3
import sys, numpy as np, pandas as pd
V = str(V3)
sys.path.insert(0, V)
sys.path.insert(0, f"{V}")
M = str(MIMIC_RAW)
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
ROUTES = ["IV", "PO/NG", "PO", "NU", "ORAL"]

# 1. Bolton rules on the raw prescriptions, without iv_duration<=7
import os
SPANS = f"{CACHE}/mimic_spans_nocap.parquet"
if os.path.exists(SPANS):
    W = pd.read_parquet(SPANS); print(f"cached {len(W):,} stay")
else:
    pat = "|".join(STEMS); keep = []
    for ch in pd.read_csv(f"{M}/hosp/prescriptions.csv",
            usecols=["hadm_id", "starttime", "stoptime", "drug", "route"],
            dtype={"hadm_id": "float64", "drug": "string", "route": "string"},
            chunksize=2_000_000, low_memory=False):
        ch = ch[ch.route.str.contains("|".join(ROUTES), na=False, case=False)]
        keep.append(ch[ch.drug.str.lower().str.contains(pat, na=False, regex=True)])
    rx = pd.concat(keep, ignore_index=True)
    rx["route2"] = rx.route.replace({"PO/NG": "PO", "NU": "PO", "ORAL": "PO"})
    rx = rx[rx.route2.isin(["IV", "PO"])].dropna(subset=["hadm_id"])
    rx["s"] = pd.to_datetime(rx.starttime, errors="coerce")
    rx["e"] = pd.to_datetime(rx.stoptime, errors="coerce")
    rx = rx.dropna(subset=["s"])
    icu = pd.read_csv(f"{M}/icu/icustays.csv", usecols=["hadm_id", "stay_id", "intime", "outtime"])
    icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
    j = rx.merge(icu, on="hadm_id", how="inner")
    j = j[(j.s >= j.intime) & (j.s <= j.outtime)]
    sp = j.groupby(["stay_id", "route2"]).agg(st=("s", "min"), sp=("e", "max")).reset_index()
    W = sp.pivot(index="stay_id", columns="route2", values=["st", "sp"])
    W.columns = ["iv_start", "po_start", "iv_stop", "po_stop"]
    W = W.dropna().reset_index()
    for c in ["iv_start", "po_start", "iv_stop", "po_stop"]:
        W[c] = pd.to_datetime(W[c]).dt.normalize()
    W["iv_dur"] = (W.iv_stop - W.iv_start).dt.days
    W["po_dur"] = (W.po_stop - W.po_start).dt.days
    W = W[(W.iv_stop <= W.po_stop) & (W.iv_dur >= 0) & (W.po_dur >= 0)]
    W.to_parquet(SPANS, index=False)
print(f"both IV and PO + iv_stop<=po_stop + duration>=0 : {len(W):,} stay")
print(f"  IV duration median {W.iv_dur.median():.0f}  p90 {W.iv_dur.quantile(.9):.0f}  max {W.iv_dur.max()}")
print(f"  iv_dur<=7 : {(W.iv_dur<=7).mean():.1%}  ({int((W.iv_dur<=7).sum()):,} stay) <- kept by Bolton")
print(f"  iv_dur>7  : {(W.iv_dur>7).mean():.1%}  ({int((W.iv_dur>7).sum()):,} stay) <- newly gained")

# 2. stay_id -> Episode_ID
from config import MIMIC_SRC
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "stay_id"]).drop_duplicates("Episode_ID")
sev["Episode_ID"] = sev.Episode_ID.astype(str)
W = W.merge(sev, on="stay_id", how="inner")
print(f"  mapped to Episode_ID {len(W):,}")

# 3. inject into raw, then run the preprocessing pipeline
from src import preprocess as pp
raw, r1 = pp.load_raw("mimic")
print(f"raw {r1['rows_kept']:,} rows / {r1['episodes']:,} episodes")
m = W.set_index("Episode_ID")
def ymd_i(s): return s.dt.strftime("%Y%m%d").astype("Int64")
raw["IV_Start_Date"] = raw.Episode_ID.astype(str).map(ymd_i(m.iv_start))
raw["IV_Stop_Date"]  = raw.Episode_ID.astype(str).map(ymd_i(m.iv_stop))
raw["PO_Start_Date"] = raw.Episode_ID.astype(str).map(ymd_i(m.po_start))
raw = raw[raw.IV_Stop_Date.notna()].copy()
print(f"after injection: raw {len(raw):,} rows / {raw.Episode_ID.nunique():,} episodes")

raw, r2 = pp.derive_ordinals(raw, "mimic"); print("  ordinals ok")
raw, r3 = pp.apply_ranges(raw);             print("  ranges ok")
raw, r4 = pp.filter_span(raw);              print(f"  span: {r4}")
per, r5 = pp.extract_periods_mimic(raw, "last_iv_plus_1")
print(f"  periods: {r5}")
meta, r6 = pp.build_anchors(raw, per, "mimic", switch_definition="last_iv_plus_1")
print(f"  anchors: {len(meta):,} / episodes {meta.Episode_ID.nunique():,}")
print(f"           delta median {meta.delta_days.median():.0f}  max {meta.delta_days.max()}")
iv = (pd.to_datetime(meta.switch_day) - pd.to_datetime(meta.Episode_ID.map(m.iv_start))).dt.days - 1
print(f"           IV duration median {iv.median():.0f}  fraction >7 {(iv>7).mean():.1%}")
arrays, r7 = pp.build_tensors(meta, raw, verbose=True)
print(f"  tensors: {r7}")
# build_tensors returns (vit, vmask, lab, lmask, static, meta); write it in the same
# layout as save(). Channel pruning happens later in load_domain via the registry.
import json as _json
from config import (WINDOW_TAG, INPUT_DAYS, BIN_HOURS, N_BINS, VITALS, LABS, STATIC,
                    ORDINAL_CHANNELS)
vit, vmask, lab, lmask, static, meta_out = arrays
OUT = f"{DATA}"
os.makedirs(OUT, exist_ok=True)
spec = dict(window_tag=WINDOW_TAG, input_days=INPUT_DAYS, bin_hours=BIN_HOURS, n_bins=N_BINS,
            vitals=VITALS, labs=LABS, static=STATIC, ordinal_channels=list(ORDINAL_CHANNELS),
            filled=False, switch_definition="last_iv_plus_1", iv_duration_cap=None,
            note="Bolton rules minus iv_duration<=7")
np.savez_compressed(f"{OUT}/mimic_tensors_{WINDOW_TAG}.npz",
    vit=vit, vit_mask=vmask, lab=lab, lab_mask=lmask, static=static,
    sample_id=meta_out["sample_id"].to_numpy(np.int64),
    episode_id=meta_out["Episode_ID"].astype(str).to_numpy(),
    vit_cols=np.array(VITALS), lab_cols=np.array(LABS), static_cols=np.array(STATIC),
    spec=_json.dumps(spec, ensure_ascii=False))
meta_out.to_parquet(f"{OUT}/mimic_meta_{WINDOW_TAG}.parquet", index=False)
print(f"\n-> {OUT}/  vit{vit.shape} lab{lab.shape}  meta {len(meta_out):,}")
