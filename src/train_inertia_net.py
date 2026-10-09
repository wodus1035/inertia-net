"""Joint KNU + MIMIC-IV training of Inertia-Net.

A shared triplet encoder produces one scalar readiness score s(z); each cohort and
horizon contributes only a cutpoint, logit[d, h] = tau * (s(z) - c[d, h]). The loss
is class-weighted BCE per cohort plus a multi-kernel MMD term aligning the two
representations. Patient-level 5-fold CV re-splits train+val; test is never touched.

    python -m src.train_inertia_net --gpu 0 --tag cv_f0_s42 --seed 42 --fold 0 --n_folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

from .config import OUTPUTS, patient_of, VITALS, LAB_COLS
from .data import load_domain, make_Y, fit_stats, fit_static, apply_stats, masks_of
from .triplets import build_triplets, normalize_triplet_values
from .model import MultiModalTransformer, WeightedBCELoss, TemperatureScaler, mmd_loss, InertiaHead


DOMAINS = ["knu", "mimic"]
MMD_SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _build(d, idx, stats, horizons):
    """Triplet tokens, static features and cumulative labels for the rows in `idx`."""
    vc, vs, lc, ls, sc, ss = stats
    trip, valid = build_triplets(d["vit"][idx], d["vmask"][idx], d["lab"][idx], d["lmask"][idx])
    vi = normalize_triplet_values(trip, np.concatenate([vc, lc]), np.concatenate([vs, ls]))
    li = valid[:, :, None]
    si = apply_stats(d["static"][idx], sc, ss)
    y = make_Y(d["delta"][idx], horizons)
    return vi, li, si, y


def _loader(arrs, batch_size, shuffle, num_workers=2):
    tens = [torch.from_numpy(a.astype(np.float32)) for a in arrs]
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tens),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _subsample_mask(mask, frac, seed):
    if frac >= 1.0:
        return mask
    idx = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(len(idx) * frac))
    keep = rng.choice(idx, n_keep, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def _make_cv_fold_masks(combined_mask, episode_ids, n_folds, seed):
    """Split `combined_mask` (train | val) into n_folds folds, one patient's
    anchor-days never crossing folds."""
    combined_idx = np.where(combined_mask)[0]
    patients = patient_of(pd.Series(episode_ids[combined_idx]).astype(str)).to_numpy()

    uniq_patients = np.array(sorted(set(patients)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq_patients)
    fold_of_patient = {p: i % n_folds for i, p in enumerate(uniq_patients)}
    row_fold = np.array([fold_of_patient[p] for p in patients])

    fold_masks = []
    for f in range(n_folds):
        m = np.zeros_like(combined_mask)
        m[combined_idx[row_fold == f]] = True
        fold_masks.append(m)
    return fold_masks


@torch.no_grad()
def eval_domain(encoder, heads, domain, loader, device, criterion):
    encoder.eval(); heads.eval()
    logits_all, y_all = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        xv, xl, s, y = (t.to(device) for t in batch[:4])
        _, fused = encoder(xv, xl, s, return_fused=True)
        logits = heads(fused, domain) * encoder.logit_scale
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        total_n += len(y)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(y_all), total_loss / total_n


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    return float(thr[best]) if best < len(thr) else 0.5


def _per_head(fn, y, p):
    out = np.full(y.shape[1], np.nan)
    for h in range(y.shape[1]):
        if len(np.unique(y[:, h])) > 1:
            out[h] = fn(y[:, h], p[:, h])
    return out


def auroc_per_head(y, p):
    return _per_head(roc_auc_score, y, p)


def auprc_per_head(y, p):
    return _per_head(average_precision_score, y, p)


def f1_per_head(y, p, thresholds):
    out = np.full(y.shape[1], np.nan)
    for h in range(y.shape[1]):
        out[h] = f1_score(y[:, h], (p[:, h] >= thresholds[h]).astype(int), zero_division=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--knu_batch_size", type=int, default=64)
    ap.add_argument("--mimic_batch_size", type=int, default=64)
    ap.add_argument("--max_grad_norm", type=float, default=5.2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--logit_scale", type=float, default=3.0, help="tau")
    ap.add_argument("--strats_fusion", action=argparse.BooleanOptionalAction, default=True)
    # KNU-only ablation: no MIMIC batches and no MMD.
    ap.add_argument("--knu_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--shared_cutpoints", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--lambda_mmd", type=float, default=0.8)
    ap.add_argument("--mmd_warmup_epochs", type=int, default=10)
    ap.add_argument("--robust", action=argparse.BooleanOptionalAction, default=True,
                    help="median/IQR normalization statistics (mean/SD otherwise)")
    # With W&B on, only scalar metrics leave the machine: code, git and system
    # statistics are disabled below (MIMIC DUA, KNU IRB).
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--wandb_project", type=str, default="jamia-inertia")
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True,
                    help="TensorBoard logging under IVOS_OUTPUTS/tb/<tag>")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample_frac", type=float, default=1.0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0, help="which fold is validation this run")
    ap.add_argument("--cv_seed", type=int, default=42, help="fold assignment seed, fixed across a sweep")
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quick_test", action="store_true", default=False)
    args = ap.parse_args()

    if args.quick_test:
        args.subsample_frac = 0.1
        args.epochs = 15
        args.patience = 5
        args.val_every = 2
        args.mmd_warmup_epochs = 3
        args.tag = args.tag or "quick_test_inertia"
        print(f"[QUICK_TEST MODE] subsample_frac={args.subsample_frac}  epochs={args.epochs}  "
              f"patience={args.patience}  val_every={args.val_every}\n")

    horizons = tuple(args.horizons)
    n_horizons = len(horizons)

    base_tag = args.tag or "run"
    args.tag = f"{base_tag}_fold{args.fold}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True   # runs with the same seed are therefore not bit-identical
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUTPUTS) / "inertia_net" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0 <= args.fold < args.n_folds):
        raise ValueError(f"--fold must be in [0, {args.n_folds}) — got {args.fold}")

    d_knu = load_domain("knu")
    knu_tr_orig, knu_va_orig, knu_te = masks_of(d_knu, "train"), masks_of(d_knu, "val"), masks_of(d_knu, "test")
    d_mimic = load_domain("mimic")
    mimic_tr_orig, mimic_va_orig, mimic_te = masks_of(d_mimic, "train"), masks_of(d_mimic, "val"), masks_of(d_mimic, "test")

    def _fold_split(combined, eps):
        folds = _make_cv_fold_masks(combined, eps, args.n_folds, args.cv_seed)
        va = folds[args.fold]
        tr = np.zeros_like(combined)
        for f in range(args.n_folds):
            if f != args.fold:
                tr |= folds[f]
        return _subsample_mask(tr, args.subsample_frac, args.seed), va

    knu_tr, knu_va = _fold_split(knu_tr_orig | knu_va_orig, d_knu["eps"])
    mimic_tr, mimic_va = _fold_split(mimic_tr_orig | mimic_va_orig, d_mimic["eps"])

    print(f"  [CV] fold {args.fold}/{args.n_folds}  (cv_seed={args.cv_seed})  "
          f"— re-split from original train+val; test untouched (KNU test={knu_te.sum():,}, MIMIC test={mimic_te.sum():,})")
    print(f"[KNU]   train {knu_tr.sum():,} | val(fold) {knu_va.sum():,} | test {knu_te.sum():,}")
    print(f"[MIMIC] train {mimic_tr.sum():,} | val(fold) {mimic_va.sum():,} | test {mimic_te.sum():,}")

    # normalization statistics come from each cohort's own training rows
    def _stats(d, tr):
        return (*fit_stats(d["vit"][tr], d["vmask"][tr], args.robust),
                *fit_stats(d["lab"][tr], d["lmask"][tr], args.robust),
                *fit_static(d["static"][tr]))

    knu_stats, mimic_stats = _stats(d_knu, knu_tr), _stats(d_mimic, mimic_tr)

    Vk_tr, Lk_tr, Sk_tr, Yk_tr = _build(d_knu, knu_tr, knu_stats, horizons)
    Vk_va, Lk_va, Sk_va, Yk_va = _build(d_knu, knu_va, knu_stats, horizons)
    Vm_tr, Lm_tr, Sm_tr, Ym_tr = _build(d_mimic, mimic_tr, mimic_stats, horizons)
    Vm_va, Lm_va, Sm_va, Ym_va = _build(d_mimic, mimic_va, mimic_stats, horizons)

    if Vk_tr.shape[-1] != Vm_tr.shape[-1] or Sk_tr.shape[-1] != Sm_tr.shape[-1]:
        raise ValueError("KNU/MIMIC feature dims don't match — the shared encoder requires identical layouts.")

    p_knu, p_mimic = Yk_tr.mean(0), Ym_tr.mean(0)
    _w = lambda p: (1 - p) / np.clip(p, 1e-6, 1)
    pw_knu = torch.tensor(_w(p_knu), dtype=torch.float32, device=device)
    pw_mimic = torch.tensor(_w(p_mimic), dtype=torch.float32, device=device)
    print(f"  KNU   pos_rate/head={np.round(p_knu,3)}   pos_weight/head={np.round(pw_knu.cpu().numpy(),2)}")
    print(f"  MIMIC pos_rate/head={np.round(p_mimic,3)} pos_weight/head={np.round(pw_mimic.cpu().numpy(),2)}")

    print("=" * 66)
    print(f"TRAIN Inertia-Net   tag={args.tag}  lambda_mmd={args.lambda_mmd}  "
          f"mmd_warmup_epochs={args.mmd_warmup_epochs}  knu_only={args.knu_only}  "
          f"shared_cutpoints={args.shared_cutpoints}")
    print("=" * 66)

    n_channels = len(VITALS) + len(LAB_COLS)
    encoder = MultiModalTransformer(
        vital_input_dim=n_channels, static_input_dim=Sk_tr.shape[-1],
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout,
        n_horizons=n_horizons, logit_scale=args.logit_scale, strats_fusion=args.strats_fusion,
    ).to(device)
    heads = InertiaHead(args.d_model, n_horizons, DOMAINS, shared_cutpoints=args.shared_cutpoints).to(device)

    criterion_knu = WeightedBCELoss(pw_knu, smoothing=args.smoothing).to(device)
    criterion_mimic = WeightedBCELoss(pw_mimic, smoothing=args.smoothing).to(device)

    params = list(encoder.parameters()) + list(heads.parameters())
    wb = None
    if args.wandb:
        import wandb as _wb
        cfg = {k: v for k, v in vars(args).items() if k != "tag"}
        wb = _wb.init(project=args.wandb_project, name=args.tag, config=cfg,
                      settings=_wb.Settings(disable_git=True, disable_code=True,
                                            save_code=False, disable_job_creation=True,
                                            x_disable_stats=True, x_disable_meta=True))
        print(f"  [wandb] {wb.url}")

    writer = None
    if args.tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(OUTPUTS) / "tb" / args.tag
            writer = SummaryWriter(str(tb_dir))
            print(f"  [tb] {tb_dir}")
        except ImportError:
            print("  [tb] tensorboard not installed — skipped")

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, threshold=1e-3, threshold_mode="rel"
    )
    amp_enabled = args.amp and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    print(f"  amp_enabled={amp_enabled}")

    knu_loader = _loader([Vk_tr, Lk_tr, Sk_tr, Yk_tr], args.knu_batch_size, True)
    mimic_loader = _loader([Vm_tr, Lm_tr, Sm_tr, Ym_tr], args.mimic_batch_size, True)
    knu_val_loader = _loader([Vk_va, Lk_va, Sk_va, Yk_va], 512, False)
    mimic_val_loader = _loader([Vm_va, Lm_va, Sm_va, Ym_va], 512, False)
    knu_iter, mimic_iter = _cycle(knu_loader), _cycle(mimic_loader)
    steps_per_epoch = max(len(knu_loader), len(mimic_loader))

    best_metric, best_state, best_epoch, no_improve = -1.0, None, 0, 0

    for ep in range(1, args.epochs + 1):
        lambda_mmd_ep = args.lambda_mmd * min(1.0, ep / max(1, args.mmd_warmup_epochs))

        encoder.train(); heads.train()
        ep_sup_k, ep_sup_m, ep_mmd, ep_n = 0.0, 0.0, 0.0, 0
        for _ in range(steps_per_epoch):
            xvk, xlk, sk, yk = (t.to(device) for t in next(knu_iter))
            xvm, xlm, sm, ym = (t.to(device) for t in next(mimic_iter))

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                _, fused_k = encoder(xvk, xlk, sk, return_fused=True)
                _, fused_m = encoder(xvm, xlm, sm, return_fused=True)
                logits_k = heads(fused_k, "knu") * encoder.logit_scale
                logits_m = heads(fused_m, "mimic") * encoder.logit_scale
                loss_k = criterion_knu(logits_k, yk)
                loss_m = criterion_mimic(logits_m, ym)
                if args.knu_only:
                    mmd = torch.zeros((), device=device)
                    loss = loss_k
                else:
                    mmd = mmd_loss(fused_k.float(), fused_m.float(), sigmas=MMD_SIGMAS)
                    loss = loss_k + loss_m + lambda_mmd_ep * mmd

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            bs = len(yk) + len(ym)
            ep_sup_k += loss_k.item() * len(yk)
            ep_sup_m += loss_m.item() * len(ym)
            ep_mmd += mmd.item() * bs
            ep_n += bs

        do_eval = (ep % args.val_every == 0) or (ep == args.epochs)
        if not do_eval:
            print(f"  ep {ep:>3d}/{args.epochs}  sup_knu={ep_sup_k/ep_n*2:.4f}  "
                  f"sup_mimic={ep_sup_m/ep_n*2:.4f}  mmd={ep_mmd/ep_n:.4f}  "
                  f"lambda_mmd={lambda_mmd_ep:.3f}  [val skipped]")
            continue

        k_logits, k_y, k_loss = eval_domain(encoder, heads, "knu", knu_val_loader, device, criterion_knu)
        m_logits, m_y, m_loss = eval_domain(encoder, heads, "mimic", mimic_val_loader, device, criterion_mimic)
        k_prob, m_prob = 1/(1+np.exp(-k_logits)), 1/(1+np.exp(-m_logits))
        k_auroc, k_auprc = auroc_per_head(k_y, k_prob), auprc_per_head(k_y, k_prob)
        m_auroc, m_auprc = auroc_per_head(m_y, m_prob), auprc_per_head(m_y, m_prob)

        # selection metric: mean validation AUROC over horizons and cohorts
        k_sel, m_sel = float(np.nanmean(k_auroc)), float(np.nanmean(m_auroc))
        sel = k_sel if args.knu_only else (k_sel + m_sel) / 2

        scheduler.step(sel)
        cur_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            ck_, cm_ = heads.cutpoints("knu").cpu().numpy(), heads.cutpoints("mimic").cpu().numpy()
        log = {"epoch": ep, "sel/both": sel, "sel/knu": k_sel, "sel/mimic": m_sel,
               "loss/val_knu": k_loss, "loss/val_mimic": m_loss,
               "loss/train_sup_knu": ep_sup_k / max(ep_n, 1),
               "loss/train_sup_mimic": ep_sup_m / max(ep_n, 1),
               "loss/mmd": ep_mmd / max(ep_n, 1), "lr": cur_lr}
        for i, h in enumerate(horizons):
            log[f"auroc_knu/h{h}"] = float(k_auroc[i]); log[f"auroc_mimic/h{h}"] = float(m_auroc[i])
            log[f"auprc_knu/h{h}"] = float(k_auprc[i]); log[f"auprc_mimic/h{h}"] = float(m_auprc[i])
            log[f"cutpoint_knu/h{h}"] = float(ck_[i]); log[f"cutpoint_mimic/h{h}"] = float(cm_[i])
            log[f"raw_gap/h{h}"] = float(ck_[i] - cm_[i])
        if writer is not None:
            for k, v in log.items():
                if k != "epoch":
                    writer.add_scalar(k, v, ep)
        if wb is not None:
            wb.log(log, step=ep)

        improved = sel > best_metric
        if improved:
            best_metric, best_epoch = sel, ep
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        print(f"  ep {ep:>3d}/{args.epochs}  sup_knu={ep_sup_k/ep_n*2:.4f}  sup_mimic={ep_sup_m/ep_n*2:.4f}  "
              f"mmd={ep_mmd/ep_n:.4f}  lambda_mmd={lambda_mmd_ep:.3f}  lr={cur_lr:.2e}\n"
              f"      KNU   AUROC={np.round(k_auroc,3)}  AUPRC={np.round(k_auprc,3)}\n"
              f"      MIMIC AUROC={np.round(m_auroc,3)}  AUPRC={np.round(m_auprc,3)}\n"
              f"      sel={sel:.4f}@best={best_metric:.4f}@{best_epoch}  "
              f"no_improve={no_improve}/{args.patience}" + ("  *" if improved else ""))

        if no_improve >= args.patience:
            print(f"  early stop at epoch {ep}")
            break

    encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
    heads.load_state_dict({k: v.to(device) for k, v in best_state["heads"].items()})

    def _final_eval(domain, loader, criterion):
        logits, y, _ = eval_domain(encoder, heads, domain, loader, device, criterion)
        scaler = TemperatureScaler(n_horizons=n_horizons)
        T = scaler.fit(torch.from_numpy(logits), torch.from_numpy(y))
        prob_cal = 1/(1+np.exp(-logits/np.array(T)))
        thr = [best_f1_threshold(y[:, h], prob_cal[:, h]) for h in range(n_horizons)]
        auroc, auprc = auroc_per_head(y, prob_cal), auprc_per_head(y, prob_cal)
        f1 = f1_per_head(y, prob_cal, thr)
        print(f"\n  [{domain} val] T={np.round(T,4)}  thr={np.round(thr,4)}")
        print(f"    AUROC={np.round(auroc,4)}  AUPRC={np.round(auprc,4)}  F1={np.round(f1,4)}")
        return T, thr, auroc.tolist(), auprc.tolist(), f1.tolist()

    T_knu, thr_knu, auroc_knu, auprc_knu, f1_knu = _final_eval("knu", knu_val_loader, criterion_knu)
    T_mimic, thr_mimic, auroc_mimic, auprc_mimic, f1_mimic = _final_eval("mimic", mimic_val_loader, criterion_mimic)

    with torch.no_grad():
        cuts_knu = heads.cutpoints("knu").cpu().numpy()
        cuts_mimic = heads.cutpoints("mimic").cpu().numpy()
    print(f"\n  [cutpoints]  KNU={np.round(cuts_knu,4)}  MIMIC={np.round(cuts_mimic,4)}  "
          f"raw difference={np.round(cuts_knu - cuts_mimic,4)}")
    print("  (the manuscript's switching-threshold gap is estimated post hoc on s(z) with "
          "scripts/sensitivity.py, not from these training cutpoints)")

    ckpt = {
        "encoder_state_dict": best_state["encoder"], "heads_state_dict": best_state["heads"],
        "horizons": list(horizons), "n_horizons": n_horizons, "domains": DOMAINS,
        "temperature_per_domain": {"knu": T_knu, "mimic": T_mimic},
        "threshold_per_domain": {"knu": thr_knu, "mimic": thr_mimic},
        "cutpoints_per_domain": {"knu": cuts_knu.tolist(), "mimic": cuts_mimic.tolist()},
        "best_epoch": int(best_epoch), "best_val_metric": float(best_metric),
        "fold": args.fold, "n_folds": args.n_folds, "cv_seed": args.cv_seed,
        "hp": {"d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
               "dropout": float(args.dropout), "logit_scale": args.logit_scale,
               "lambda_mmd": args.lambda_mmd, "vital_input_dim": n_channels,
               "static_input_dim": int(Sk_tr.shape[-1]), "seq_encoder": "strats",
               "strats_fusion": args.strats_fusion, "shared_cutpoints": args.shared_cutpoints},
        "norm_stats_per_domain": {
            "knu": dict(zip(["vc","vs","lc","ls","sc","ss"], knu_stats)),
            "mimic": dict(zip(["vc","vs","lc","ls","sc","ss"], mimic_stats)),
        },
        "config": vars(args),
        "val_metrics": {
            "knu": {"auroc": auroc_knu, "auprc": auprc_knu, "f1": f1_knu},
            "mimic": {"auroc": auroc_mimic, "auprc": auprc_mimic, "f1": f1_mimic},
        },
    }
    if wb is not None:
        wb.summary.update({"best_sel": float(best_metric), "best_epoch": int(best_epoch),
                           "knu_auroc_h1": float(auroc_knu[0]), "mimic_auroc_h1": float(auroc_mimic[0])})
        wb.finish()
    if writer is not None:
        writer.close()
    torch.save(ckpt, out_dir / "inertia_net_model.pth")
    with open(out_dir / "inertia_net_report.json", "w") as f:
        json.dump({"tag": args.tag, "horizons": list(horizons), "best_epoch": int(best_epoch),
                   "config": vars(args), "val_metrics": ckpt["val_metrics"],
                   "cutpoints_per_domain": ckpt["cutpoints_per_domain"]}, f, indent=2)
    print(f"\n  saved -> {out_dir/'inertia_net_model.pth'}")
    print(f"RESULT tag={args.tag} best_epoch={best_epoch} "
          f"KNU_auroc={np.round(auroc_knu,4).tolist()} MIMIC_auroc={np.round(auroc_mimic,4).tolist()}")


if __name__ == "__main__":
    main()
