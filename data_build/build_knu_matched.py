"""Build the KNU cohort under the same rules as the no-cap MIMIC cohort.

Per episode: one IV and one PO span, both required, iv_stop <= po_stop, duration >= 0,
no cap on IV duration, label = iv_stop + 1. The unit of analysis still differs
(MIMIC = ICU stay, KNU = admission); that is noted in the Limitations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.common import CACHE, DATA, V3
import sys, os, json, numpy as np, pandas as pd
V = str(V3)
sys.path.insert(0, V)
from src import preprocess as pp
from config import (WINDOW_TAG, INPUT_DAYS, BIN_HOURS, N_BINS, VITALS, LABS, STATIC,
                    ORDINAL_CHANNELS)

raw, r1 = pp.load_raw("knu")
print(f"raw {r1['rows_kept']:,} rows / {r1['episodes']:,} episodes")
raw, _ = pp.derive_ordinals(raw, "knu")
raw, _ = pp.apply_ranges(raw)
raw, r4 = pp.filter_span(raw); print(f"  span: {r4}")

# spans under the same rules as MIMIC
dr = raw[["Episode_ID", "Event_Date", "Drug_Codes"]].dropna()
def has(s, pre): return any(c.strip().upper().startswith(pre) for c in str(s).split("|") if c.strip())
dr = dr.assign(iv=dr.Drug_Codes.map(lambda s: has(s, "DI")),
               po=dr.Drug_Codes.map(lambda s: has(s, "DO")))
day = dr.groupby(["Episode_ID", "Event_Date"], as_index=False).agg(iv=("iv","any"), po=("po","any"))
def ymd(s): return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
day["d"] = ymd(day.Event_Date)
a = day[day.iv].groupby("Episode_ID").d.agg(iv_start="min", iv_stop="max")
b = day[day.po].groupby("Episode_ID").d.agg(po_start="min", po_stop="max")
W = a.join(b, how="inner").reset_index()
n0 = len(W)
W["iv_dur"] = (W.iv_stop - W.iv_start).dt.days
W["po_dur"] = (W.po_stop - W.po_start).dt.days
W = W[(W.iv_stop <= W.po_stop) & (W.iv_dur >= 0) & (W.po_dur >= 0)]
print(f"\nKNU: both IV and PO {n0:,}  ->  passing iv_stop<=po_stop & duration>=0 {len(W):,}")
print(f"  IV duration median {W.iv_dur.median():.0f}  p90 {W.iv_dur.quantile(.9):.0f}  max {W.iv_dur.max()}")
print(f"  iv_dur<=7 {(W.iv_dur<=7).mean():.1%}   >7 {(W.iv_dur>7).mean():.1%}   <- no cap")

st = raw.groupby("Episode_ID").agg(Sex=("Sex","first"), Age=("Age","first")).reset_index()
per = pd.DataFrame({"Episode_ID": W.Episode_ID, "iv_start": W.iv_start, "last_iv": W.iv_stop,
                    "first_po": W.po_start,
                    "switch_day": W.iv_stop + pd.Timedelta(days=1),
                    "reesc_day": pd.NaT}).merge(st, on="Episode_ID", how="left")
print(f"  periods {len(per):,}")

meta, r6 = pp.build_anchors(raw, per, "knu", switch_definition="last_iv_plus_1")
print(f"  anchors {len(meta):,} / episodes {meta.Episode_ID.nunique():,}"
      f"  delta median {meta.delta_days.median():.0f} max {meta.delta_days.max()}")
arrays, r7 = pp.build_tensors(meta, raw, verbose=True)
vit, vmask, lab, lmask, static, meta_out = arrays
OUT = f"{DATA}"; os.makedirs(OUT, exist_ok=True)
spec = dict(window_tag=WINDOW_TAG, input_days=INPUT_DAYS, bin_hours=BIN_HOURS, n_bins=N_BINS,
            vitals=VITALS, labs=LABS, static=STATIC, ordinal_channels=list(ORDINAL_CHANNELS),
            filled=False, switch_definition="last_iv_plus_1", iv_duration_cap=None,
            note="same rules as MIMIC nocap: single IV/PO span, iv_stop<=po_stop, duration>=0")
np.savez_compressed(f"{OUT}/knu_tensors_{WINDOW_TAG}.npz",
    vit=vit, vit_mask=vmask, lab=lab, lab_mask=lmask, static=static,
    sample_id=meta_out["sample_id"].to_numpy(np.int64),
    episode_id=meta_out["Episode_ID"].astype(str).to_numpy(),
    vit_cols=np.array(VITALS), lab_cols=np.array(LABS), static_cols=np.array(STATIC),
    spec=json.dumps(spec, ensure_ascii=False))
meta_out.to_parquet(f"{OUT}/knu_meta_{WINDOW_TAG}.parquet", index=False)
print(f"\n-> {OUT}/knu_*  vit{vit.shape}  meta {len(meta_out):,}")
