import numpy as np
import pandas as pd

import os
import sys

from .config import (MIMIC_TENS, KNU_TENS, MIMIC_META, KNU_META, STATIC_NUMERIC_IDX)
from .clip_registry import clip_domain_by_registry, VITAL_REGISTRY, LAB_REGISTRY
from .split import sample_split




def load_domain(domain):
    """domain ('mimic | 'knu') dict(vit, vmask, lab, lmask, static, eps, delta, split).

    Args:
        domain (str): 'mimic' or 'knu' or 'external'
    """
    
    tens = MIMIC_TENS if domain == "mimic" else KNU_TENS
    metap = MIMIC_META if domain == "mimic" else KNU_META
    
    z = np.load(str(tens), allow_pickle=True)
    m = pd.read_parquet(str(metap))
    eps = z["episode_id"].astype(str)

    vit = clip_domain_by_registry(z["vit"], z["vit_cols"], VITAL_REGISTRY)
    lab = clip_domain_by_registry(z["lab"], z["lab_cols"], LAB_REGISTRY)

    # Align meta to the npz arrays explicitly by sample_id and cross-check with
    # episode_id: positional alignment would fail silently and mislabel samples.
    sid = z["sample_id"]
    if len(m) != len(sid):
        raise ValueError(f"[load_domain:{domain}] meta rows({len(m)}) != npz samples({len(sid)})")
    m_by_sid = m.set_index("sample_id")
    missing = set(sid.tolist()) - set(m_by_sid.index.tolist())
    if missing:
        raise ValueError(f"[load_domain:{domain}] {len(missing)} npz sample_id missing in meta, e.g. {list(missing)[:5]}")
    m_aligned = m_by_sid.loc[sid]
    delta = m_aligned["delta_days"].to_numpy()
    meta_eps = m_aligned["Episode_ID"].astype(str).to_numpy()
    if not np.array_equal(meta_eps, eps):
        n_mismatch = int((meta_eps != eps).sum())
        raise ValueError(
            f"[load_domain:{domain}] episode_id mismatch on {n_mismatch} samples after "
            f"sample_id alignment — npz tensors and meta are not the same cohort/order.")

    # Course day (days since IV start). meta stores anchor -> IV start, so flip the
    # sign; older datasets lack the column and get NaN.
    course = (-m_aligned["delta_iv_start"].to_numpy(dtype=np.float32)
              if "delta_iv_start" in m_aligned.columns
              else np.full(len(eps), np.nan, dtype=np.float32))

    return dict(
        vit=vit, vmask=z["vit_mask"], lab=lab, lmask=z["lab_mask"],
        static=z["static"], eps=eps,
        delta=delta,
        course=course,
        split=sample_split(domain, eps),
    )




def make_Y(delta, horizons):
    return (delta[:, None] <= np.array(horizons)[None, :]).astype(np.float32)

def fit_stats(X, M, robust=False):
    """Per-channel (center, scale) over measured cells only; robust=True uses median/IQR."""
    F = X.shape[-1]
    Xf = X.reshape(-1, F); Mf = M.reshape(-1, F).astype(bool)
    c = np.zeros(F, np.float32); s = np.ones(F, np.float32)
    for f in range(F):
        v = Xf[Mf[:, f], f]; v = v[~np.isnan(v)]
        if v.size > 1:
            if robust:
                c[f] = np.median(v)
                q75, q25 = np.percentile(v, [75, 25]); r = q75 - q25
                s[f] = r if r > 1e-6 else 1.0
            else:
                c[f] = v.mean(); st = v.std(); s[f] = st if st > 1e-6 else 1.0
    return c, s
 
 
def apply_stats(X, c, s):
    Xs = (X - c) / s
    Xs[np.isnan(Xs)] = 0.0
    return Xs.astype(np.float32)
 
 
def fit_static(S):
    c = np.zeros(S.shape[1], np.float32); s = np.ones(S.shape[1], np.float32)
    for i in STATIC_NUMERIC_IDX:
        v = S[:, i]; v = v[~np.isnan(v)]
        if v.size > 1:
            c[i] = v.mean(); st = v.std(); s[i] = st if st > 1e-6 else 1.0
    return c, s
 
 

def masks_of(d, which):
    """Boolean mask of samples whose split == which."""
    return d["split"] == which