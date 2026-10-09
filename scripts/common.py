"""Shared paths, checkpoint loading, s(z) scoring and statistics for the analysis scripts.

All paths default to locations inside the repository and can be overridden through the
IVOS_DATAS / IVOS_MODELS environment variables. Raw-source locations come from
src/local_paths.py (git-ignored) or the matching environment variable.
"""
import os, sys
from pathlib import Path

import numpy as np

def _raw(name):
    """Raw-source location: environment variable, then src/local_paths.py (git-ignored)."""
    try:
        from src.local_paths import PATHS as _LOCAL
    except ImportError:
        _LOCAL = {}
    return os.environ.get(name) or _LOCAL.get(name) or f"<set {name}>"


def _dir(env, default):
    """Use the env var only if it points at something that exists; else the repo default."""
    v = os.environ.get(env)
    if v and Path(v).exists():
        return Path(v)
    return Path(default)


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = _dir("IVOS_DATAS", REPO / "data" / "ds_nocap")
MODELS = _dir("IVOS_MODELS", REPO / "models")
RESULTS = REPO / "results"
LOGS = REPO / "logs"
PAPER = Path(os.environ.get("IVOS_PAPER", REPO.parent))   # manuscript tree (submission/JAMIA)
CACHE = REPO / "data" / "cache"      # regenerable derivatives, not tracked
CACHE.mkdir(parents=True, exist_ok=True)
FIGURES = REPO / "figures"
EICU = Path(_raw("IVOS_EICU"))
EICU_TENS = EICU / "inputs" / "nocut" / "72h" / "eicu_tensors_d3_b36x2h.npz"
V3 = Path(_raw("IVOS_V3"))                 # preprocessing tree the cohort builders read
MIMIC_RAW = Path(_raw("MIMIC_RAW"))        # PhysioNet MIMIC-IV v3.1 (hosp/, icu/)

# Drop stale raw-data env vars: a dead path in the shell would override local_paths.
for _k in ("KNU_SRC", "MIMIC_SRC"):
    _v = os.environ.get(_k)
    if _v and not Path(_v).exists():
        os.environ.pop(_k)

os.environ.setdefault("IVOS_INPUT_DAYS", "3")      # 72 h main window; set 1/2 for the 24/48 h sensitivity runs
os.environ["IVOS_DATAS"] = str(DATA)                       # assign, not setdefault: a dead value must be overwritten
_sp = DATA / "split_patients.parquet"
if _sp.exists() or not os.environ.get("IVOS_SPLIT_FILE"):
    os.environ["IVOS_SPLIT_FILE"] = str(_sp)

SEEDS = (42, 43, 44, 45)          # four seeds used for fold-0 exploration (nc_s*)
FOLDS = (0, 1, 2, 3, 4)
CV_SEEDS = (42, 43, 44)
# Reporting unit: 5 folds x 3 seeds = 15 models. Training is non-deterministic, so the
# 4-seed SD understates the true spread.
CV_TAGS = tuple(f"cv_f{f}_s{s}" for f in FOLDS for s in CV_SEEDS)
HORIZONS = (1, 2, 3)
DOMAINS = ("knu", "mimic")


def ckpt_path(tag, seed=None):
    """models/<tag>/inertia_net_model.pth; with seed given, <tag>_s<seed>."""
    name = f"{tag}_s{seed}" if seed is not None else tag
    p = MODELS / name / "inertia_net_model.pth"
    if not p.exists():                        # also accept the training output directory layout
        alt = list((MODELS / name).glob("**/inertia_net_model.pth"))
        if alt:
            return alt[0]
    return p


def load_ckpt(path, device="cuda:0"):
    """Checkpoint -> (encoder, head, ck), both in eval mode."""
    import torch
    from src.model import MultiModalTransformer, InertiaHead
    dev = torch.device(device)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    h = dict(ck["hp"])
    if h.get("seq_encoder") == "strats" and h["vital_input_dim"] == 3:
        from src.config import VITALS, LAB_COLS       # fix triplet channel count of old checkpoints
        h["vital_input_dim"] = len(VITALS) + len(LAB_COLS)
    enc = MultiModalTransformer(
        vital_input_dim=h["vital_input_dim"], static_input_dim=h["static_input_dim"],
        d_model=h["d_model"], nhead=h["nhead"], num_layers=h["num_layers"], dropout=h["dropout"],
        n_horizons=ck["n_horizons"], logit_scale=h["logit_scale"],
        strats_fusion=h.get("strats_fusion", True)).to(dev)
    # Old checkpoints carry unused classifier weights inside the encoder, hence strict=False
    enc.load_state_dict(ck["encoder_state_dict"], strict=False); enc.eval()
    hd = InertiaHead(h["d_model"], ck["n_horizons"], ck["domains"],
                     shared_cutpoints=h.get("shared_cutpoints", False)).to(dev)
    hd.load_state_dict(ck["heads_state_dict"]); hd.eval()
    return enc, hd, ck


def score(enc, hd, vit, vmask, lab, lmask, static, stats,
          device="cuda:0", batch=512):
    """Shared readiness score s(z). With stats=None, normalisation is fit on this data (labels unused)."""
    import torch
    from src.data import fit_stats, fit_static, apply_stats
    from src.triplets import build_triplets, normalize_triplet_values
    dev = torch.device(device)
    if stats is None:
        vc, vs = fit_stats(vit, vmask, robust=True)
        lc, ls = fit_stats(lab, lmask, robust=True)
        sc, ss = fit_static(static)
        stats = dict(vc=vc, vs=vs, lc=lc, ls=ls, sc=sc, ss=ss)
    trip, valid = build_triplets(vit, vmask, lab, lmask)
    vi = normalize_triplet_values(trip, np.concatenate([stats["vc"], stats["lc"]]),
                                  np.concatenate([stats["vs"], stats["ls"]]))
    S = apply_stats(static.copy()[:, None, :], stats["sc"], stats["ss"])[:, 0, :]
    out = []
    with torch.no_grad():
        for i in range(0, len(vi), batch):
            _, f = enc(torch.tensor(vi[i:i + batch]).to(dev),
                       torch.tensor(valid[i:i + batch, :, None]).to(dev),
                       torch.tensor(S[i:i + batch]).float().to(dev), return_fused=True)
            out.append(hd.score_head(f).cpu().numpy().ravel())
    return np.concatenate(out), stats


def score_split(path, domain, split="test", device="cuda:0", enc=None, hd=None, ck=None):
    """s(z) plus labels and identifiers for one split of one domain.

    Returns dict(s, Y (N,3), eps, pat, delta, stats).
    """
    import pandas as pd
    from src.data import load_domain, masks_of, make_Y
    from src.config import patient_of
    if enc is None:
        enc, hd, ck = load_ckpt(path, device)
    d = load_domain(domain)
    idx = np.where(masks_of(d, split))[0] if split else np.arange(len(d["eps"]))
    st = ck["norm_stats_per_domain"][domain]
    s, _ = score(enc, hd, d["vit"][idx], d["vmask"][idx], d["lab"][idx],
                 d["lmask"][idx], d["static"][idx], st, device)
    return dict(s=s, Y=make_Y(d["delta"][idx], HORIZONS), eps=d["eps"][idx],
                delta=d["delta"][idx], stats=st,
                pat=patient_of(pd.Series(d["eps"][idx])).to_numpy())


def score_models(tags, domains=DOMAINS, split="test", device="cuda:0", verbose=True):
    """Model tags x domains. Returns (S[dom] -> one s per model, D[dom] -> label dict)."""
    S, D = {}, {}
    for t in tags:
        enc, hd, ck = load_ckpt(ckpt_path(t), device)
        for dom in domains:
            r = score_split(None, dom, split, device, enc, hd, ck)
            S.setdefault(dom, []).append(r["s"])
            D.setdefault(dom, r)
        if verbose:
            print(f"  {t}", flush=True)
    return S, D


def score_seeds(tag, domains=DOMAINS, seeds=SEEDS, split="test", device="cuda:0",
                verbose=True):
    """Backwards-compatible wrapper: several seeds of one tag, delegating to score_models."""
    return score_models([f"{tag}_s{sd}" for sd in seeds], domains, split, device, verbose)


def c50(s, y, min_slope=0.15):
    """The s at which the observed switch probability reaches 50%: -b0/b1 of an unweighted logistic."""
    import statsmodels.api as sm
    s = np.asarray(s, float); y = np.asarray(y, float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    try:
        r = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    except Exception:
        return np.nan
    return -r.params[0] / r.params[1] if r.params[1] > min_slope else np.nan


def place(s, y, groups=None, n_boot=400, seed=0):
    """c50 with delta-method SE and, when groups (episode or patient ids) are given,
    a cluster bootstrap CI -- anchor-days are not independent.
    """
    import statsmodels.api as sm
    s = np.asarray(s, float); y = np.asarray(y, float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return dict(ok=False, reason=f"too few events: {int(y.sum())} pos / {int((1 - y).sum())} neg")
    r = sm.Logit(y, sm.add_constant(s)).fit(disp=0)
    b0, b1 = r.params; V = r.cov_params()
    if b1 <= 0.15:
        return dict(ok=False, reason=f"slope {b1:.3f} -- s does not discriminate at this site")
    c = -b0 / b1
    var = V[0, 0] / b1 ** 2 + b0 ** 2 * V[1, 1] / b1 ** 4 - 2 * b0 * V[0, 1] / b1 ** 3
    out = dict(ok=True, c50=float(c), slope=float(b1), se_delta=float(np.sqrt(max(var, 0))),
               n=len(y), n_pos=int(y.sum()))
    if groups is not None:
        bs = cluster_boot(lambda i: c50(s[i], y[i]), groups, n_boot, seed)
        bs = bs[np.isfinite(bs)]
        if len(bs) > 20:
            out.update(se_boot=float(bs.std()),
                       ci95=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))
    return out


def cluster_boot(fn, groups, n_boot=400, seed=0):
    """Cluster (episode/patient) bootstrap; fn takes an index array."""
    groups = np.asarray(groups)
    uq, inv = np.unique(groups, return_inverse=True)
    where = [np.where(inv == k)[0] for k in range(len(uq))]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uq), len(uq))
        idx = np.concatenate([where[k] for k in pick])
        try:
            out.append(fn(idx))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, float)


def gap(place_new, place_ref):
    """Difference in c50 between two sites, with SE, CI and z."""
    if not (place_new.get("ok") and place_ref.get("ok")):
        return dict(ok=False)
    g = place_new["c50"] - place_ref["c50"]
    se = float(np.hypot(place_new.get("se_boot", place_new["se_delta"]),
                        place_ref.get("se_boot", place_ref["se_delta"])))
    return dict(ok=True, gap=float(g), se=se,
                ci95=(g - 1.96 * se, g + 1.96 * se), z=float(g / se) if se else np.nan)


def mean_sd(v, nd=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f} ±{v.std():.{nd}f}"


def rule(title="", width=88):
    print(f"\n{'=' * width}\n{title}\n{'=' * width}" if title else "=" * width)
