"""Convert dense (bin x channel) matrices into a set of observed-cell triplets.

STraTS-style (Tipirneni & Reddy, TKDD 2022): a timestep is represented as
(time, channel, value), so unobserved cells simply have no token and nothing is
imputed. Aggregation into 2h/daily bins still comes from the upstream tensors.

    t = hours before the anchor; vital bin b -> (n_bins-1-b)*BIN_HOURS,
        lab day d -> (n_days-1-d)*24
    f = channel index (vitals 0..V-1, then labs V..V+L-1)
    v = raw value; normalization is applied later with fold statistics
"""

import numpy as np

PAD_FEAT = -1


def build_triplets(vit, vmask, lab, lmask, bin_hours=2, max_triplets=384, chunk=4096):
    """(N,Tv,V) + (N,Tl,L) -> (N, max_triplets, 3) [t, f, v] and (N, max_triplets) valid.

    Truncation keeps the most recent observations. Everything is float32; the
    channel index is cast to long in the model.
    """
    N, Tv, V = vit.shape
    _, Tl, L = lab.shape
    n_cell = Tv * V + Tl * L
    # For short windows n_cell can be below the cap; without lowering it,
    # order[:, :max_triplets] yields only n_cell columns and will not broadcast.
    max_triplets = min(max_triplets, n_cell)

    # (time, channel) coordinates are sample-independent, so build them once
    bv, cv = np.meshgrid(np.arange(Tv), np.arange(V), indexing="ij")
    bl, cl = np.meshgrid(np.arange(Tl), np.arange(L), indexing="ij")
    t_cell = np.concatenate([((Tv - 1 - bv) * bin_hours).ravel(),
                             ((Tl - 1 - bl) * 24).ravel()]).astype(np.float32)
    f_cell = np.concatenate([cv.ravel(), (cl + V).ravel()]).astype(np.float32)

    out = np.zeros((N, max_triplets, 3), dtype=np.float32)
    valid = np.zeros((N, max_triplets), dtype=np.float32)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        m = np.concatenate([(vmask[i:j] > 0).reshape(j - i, -1),
                            (lmask[i:j] > 0).reshape(j - i, -1)], 1)
        x = np.concatenate([np.nan_to_num(vit[i:j], nan=0.0).reshape(j - i, -1),
                            np.nan_to_num(lab[i:j], nan=0.0).reshape(j - i, -1)], 1)
        # observed cells first, then most recent (smallest t) first
        key = np.where(m, t_cell[None, :], np.float32(1e6))
        order = np.argsort(key, axis=1, kind="stable")[:, :max_triplets]
        r = np.arange(j - i)[:, None]
        sel_m = np.take_along_axis(m, order, 1)
        out[i:j, :, 0] = np.take_along_axis(key, order, 1) * sel_m
        out[i:j, :, 1] = np.where(sel_m, f_cell[order], PAD_FEAT)
        out[i:j, :, 2] = np.take_along_axis(x, order, 1) * sel_m
        valid[i:j] = sel_m.astype(np.float32)
    return out, valid


def normalize_triplet_values(trip, centers, scales):
    """Normalize raw values with fold statistics; centers/scales are in channel order."""
    out = trip.copy()
    f = out[:, :, 1].astype(np.int64)
    ok = f >= 0
    fi = np.clip(f, 0, len(centers) - 1)
    out[:, :, 2] = np.where(ok, (out[:, :, 2] - centers[fi]) / scales[fi], 0.0)
    return out
