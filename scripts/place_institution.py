"""Place a new site on the inertia axis without retraining: the score head is shared
across domains, so only the site cutpoint is fitted (two-parameter logistic on frozen s).
Prints c50, slope and the gap against a reference cohort.

Needs tensors in the usual format (vital (N,36,11), lab (N,3,15), static (N,2) with masks)
plus delta_days per anchor; roughly 400 episodes gives an acceptable c50 SE. The s axis has
no absolute scale, so the encoder must be held fixed if sites are to be compared.

usage:
    python place_institution.py --ckpt <model.pth> --tensors <site.npz> [--meta <site.parquet>]
"""
import argparse

import numpy as np

from common import DOMAINS, HORIZONS, ckpt_path, gap, load_ckpt, place, rule, score, score_split


def shared_score(ckpt, arrays, stats=None, device="cuda:0", batch=512):
    """Shared readiness score s(z) for a new site; stats=None uses the site's own robust stats."""
    enc, hd, ck = load_ckpt(ckpt, device)
    return score(enc, hd, arrays["vit"], arrays["vmask"], arrays["lab"], arrays["lmask"],
                 arrays["static"], stats, device, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: models/nc_s42")
    ap.add_argument("--tensors", required=True, help="npz of the new site")
    ap.add_argument("--meta", default=None, help="parquet holding delta_days and Episode_ID")
    ap.add_argument("--horizon", type=int, default=1, choices=HORIZONS)
    ap.add_argument("--ref", default="mimic", choices=DOMAINS, help="reference cohort for the gap")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import pandas as pd
    ck = a.ckpt or ckpt_path("nc", 42)
    z = np.load(a.tensors, allow_pickle=True)
    arrays = dict(vit=z["vit"], vmask=z["vit_mask"], lab=z["lab"],
                  lmask=z["lab_mask"], static=z["static"])
    s, _ = shared_score(ck, arrays, device=a.device)

    meta = pd.read_parquet(a.meta) if a.meta else pd.DataFrame(
        dict(delta_days=z["delta_days"], Episode_ID=z["Episode_ID"]))
    y = (meta.delta_days.to_numpy() <= a.horizon).astype(float)
    new = place(s, y, groups=meta.Episode_ID.to_numpy())

    rule(f"new site - h{a.horizon}")
    if not new["ok"]:
        print(f"  cannot place: {new['reason']}")
        return
    ci = new.get("ci95")
    print(f"  c50 {new['c50']:.3f}   slope {new['slope']:.3f}   n {new['n']:,} "
          f"(positive {new['n_pos']:,})")
    print(f"  SE  delta method {new['se_delta']:.3f}" +
          (f"   bootstrap {new['se_boot']:.3f}   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
           if ci else ""))

    r = score_split(ck, a.ref, "test", a.device)
    ref = place(r["s"], r["Y"][:, a.horizon - 1], groups=r["eps"])
    g = gap(new, ref)
    rule(f"gap - new site minus {a.ref.upper()}")
    print(f"  {a.ref.upper()} c50 {ref['c50']:.3f}")
    print(f"  gap {g['gap']:+.3f} ± {g['se']:.3f}   "
          f"95% CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]   z {g['z']:+.2f}")


if __name__ == "__main__":
    main()
