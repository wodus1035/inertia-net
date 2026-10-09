"""Patient-level split for the no-cap dataset.

Inherits assignments from the previous split wherever possible and assigns only new
patients 70/15/15, so no patient that used to be in test can leak into training.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.common import CACHE, DATA, V3
import sys, numpy as np, pandas as pd
from src.config import patient_of

# IVOS_PREV_SPLIT points at the previous split_patients.parquet; if absent, every
# patient is assigned fresh.
_prev = os.environ.get("IVOS_PREV_SPLIT", f"{CACHE}/ds_ivstop/split_patients.parquet")
old = pd.read_parquet(_prev) if os.path.exists(_prev) else pd.DataFrame(
    columns=["Patient_ID", "split"])
print(f"previous split: {len(old):,} rows  {old.domain.value_counts().to_dict()}")
prev = {(r.domain, r.patient): r.split for r in old.itertuples()}

rows = []
rng = np.random.RandomState(20260905)
for dom in ["knu", "mimic"]:
    m = pd.read_parquet(f"{DATA}/{dom}_meta_d3_b36x2h.parquet", columns=["Episode_ID"])
    eps = m.Episode_ID.astype(str).unique()
    pats = np.unique(patient_of(pd.Series(eps)).to_numpy())
    inherited = {p: prev[(dom, p)] for p in pats if (dom, p) in prev}
    newp = np.array(sorted(set(pats) - set(inherited)))
    rng.shuffle(newp)
    n = len(newp); a, b = int(n*0.70), int(n*0.85)
    assign = dict(inherited)
    for i, p in enumerate(newp):
        assign[p] = "train" if i < a else ("val" if i < b else "test")
    cnt = pd.Series(list(assign.values())).value_counts().to_dict()
    print(f"[{dom}] patients {len(pats):,}  inherited {len(inherited):,} ({len(inherited)/len(pats):.0%})  "
          f"new {n:,}  ->  {cnt}")
    pmap = pd.Series(assign)
    ap = patient_of(m.Episode_ID.astype(str)).map(pmap)
    print(f"       per anchor: {ap.value_counts(normalize=True).round(3).to_dict()}")
    for p, s in assign.items(): rows.append(dict(domain=dom, patient=p, split=s))
sp = pd.DataFrame(rows)
sp.to_parquet(f"{DATA}/split_patients.parquet", index=False)
print(f"\n-> {DATA}/split_patients.parquet  ({len(sp):,} rows)")

# leakage check: did any previously-test patient move into train/val
bad = 0
for r in sp.itertuples():
    k = (r.domain, r.patient)
    if k in prev and prev[k] == "test" and r.split != "test": bad += 1
print(f"patients moved from previous test into train/val: {bad}  (must be 0)")
