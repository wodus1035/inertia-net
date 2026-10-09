"""Redraw the manuscript figures from the no-cap cohort and the 15 CV models.

Held-out test scores are averaged over the 15 models once and cached in
results/scores15_test.npz. Palette and marks come from figures/figure_style.py.
"""
import sys, os
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, "figures"), os.path.join(_R, "scripts"), _R]
import re, sys, json
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figure_style as fs
from common import CV_SEEDS, CV_TAGS, DATA, DOMAINS, FOLDS, MIMIC_RAW, REPO, ckpt_path, load_ckpt, score_split
from src.config import KNU_SRC, MIMIC_SRC
from src.data import load_domain, masks_of
from src.train_inertia_net import _make_cv_fold_masks

PAPER = REPO.parent / "paper"; OUT = PAPER / "Figures"; OUT.mkdir(exist_ok=True); (OUT / "Sup").mkdir(exist_ok=True)
CACHE = REPO / "results" / "scores15_test.npz"; RE = REPO / "results" / "REESCALATION_NOCAP"
fs.apply_style()
def _save_both(fig, path, dpi=600):
    from pathlib import Path as _P
    fig.savefig(path, dpi=dpi); fig.savefig(_P(str(path)).with_suffix('.png'), dpi=130); fig.savefig(_P(str(path)).with_suffix('.svg')); plt.close(fig); print(f'  saved -> {path} (+png +svg)')
fs.save = _save_both; rng = np.random.default_rng(0); NB = 300

# Score cache: 15-model test scores plus a per-model Platt calibration
def build_cache():
    D = {d: load_domain(d) for d in DOMAINS}
    comb = {d: masks_of(D[d], "train") | masks_of(D[d], "val") for d in DOMAINS}
    te = {d: np.where(masks_of(D[d], "test"))[0] for d in DOMAINS}
    folds = {d: _make_cv_fold_masks(comb[d], D[d]["eps"], 5, 42) for d in DOMAINS}
    out = {}
    S = {d: [] for d in DOMAINS}; P = {d: [] for d in DOMAINS}      # test scores, Platt-prob (a,b on fold-val)
    for f in FOLDS:
        for sd in CV_SEEDS:
            enc, hd, ck = load_ckpt(ckpt_path(f"cv_f{f}_s{sd}"), "cuda:0")
            for d in DOMAINS:
                r = score_split(None, d, None, "cuda:0", enc, hd, ck); s, Y = r["s"], r["Y"]
                va = folds[d][f]; S[d].append(s[te[d]]); pr = []
                for h in range(3):
                    m = sm.Logit(Y[va, h], sm.add_constant(s[va])).fit(disp=0); a, b = m.params[1], m.params[0]
                    pr.append(1 / (1 + np.exp(-(a * s[te[d]] + b))))
                P[d].append(np.stack(pr, 1))
            print(f"  cv_f{f}_s{sd}", flush=True)
    for d in DOMAINS:
        r = score_split(None, d, "test", "cpu", *load_ckpt(ckpt_path("cv_f0_s42"), "cpu")) if False else None
        out[f"{d}_s"] = np.array(S[d]); out[f"{d}_p"] = np.array(P[d])
        out[f"{d}_Y"] = D[d]["delta"][te[d]][:, None] <= np.array([1, 2, 3])[None, :]
        out[f"{d}_eps"] = np.asarray(D[d]["eps"])[te[d]].astype(str)
        from src.config import patient_of
        out[f"{d}_pat"] = patient_of(pd.Series(out[f"{d}_eps"])).to_numpy().astype(str)
        meta = pd.read_parquet(f"{DATA}/{d}_meta_d3_b36x2h.parquet", columns=["Episode_ID", "anchor"]).iloc[te[d]]
        out[f"{d}_anchor"] = pd.to_datetime(meta.anchor).dt.normalize().astype("int64").to_numpy()
    np.savez_compressed(CACHE, **out); print(f"cache -> {CACHE}")

def load_cache():
    z = np.load(CACHE, allow_pickle=True); C = {}
    for d in DOMAINS:
        C[d] = dict(s=z[f"{d}_s"].mean(0), S=z[f"{d}_s"], p=z[f"{d}_p"].mean(0), Y=z[f"{d}_Y"].astype(float),
                    eps=z[f"{d}_eps"], pat=z[f"{d}_pat"], anchor=pd.to_datetime(z[f"{d}_anchor"]))
    return C

def icu_flags(C):
    mm = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "charttime", "icu_flag"]); mm["Episode_ID"] = mm.Episode_ID.astype(str)
    t = {"mimic": mm.assign(d=pd.to_datetime(mm.charttime).dt.normalize()).groupby(["Episode_ID", "d"]).icu_flag.max()}
    kk = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Ward", "ICU"]); kk["Episode_ID"] = kk.Episode_ID.astype(str)
    kk["d"] = pd.to_datetime(kk.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    t["knu"] = kk.assign(v=kk.ICU.notna()).groupby(["Episode_ID", "d"]).v.max()
    return {d: t[d].reindex(pd.MultiIndex.from_arrays([C[d]["eps"], C[d]["anchor"]])).fillna(0).to_numpy().astype(bool) for d in DOMAINS}

def pboot(y, pat, nb=NB):
    """95% CI of a proportion by patient-cluster bootstrap."""
    if len(y) == 0: return np.nan, np.nan
    uq, inv = np.unique(pat, return_inverse=True); where = [np.where(inv == k)[0] for k in range(len(uq))]
    bs = [y[np.concatenate([where[k] for k in rng.integers(0, len(uq), len(uq))])].mean() for _ in range(nb)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def decile_panel(ax, C, h, mask=None, title=None):
    sk, smm = C["knu"]["s"], C["mimic"]["s"]; mk = np.ones(len(sk), bool) if mask is None else mask["knu"]; mmk = np.ones(len(smm), bool) if mask is None else mask["mimic"]
    edges = np.quantile(np.r_[sk[mk], smm[mmk]], np.linspace(0, 1, 11))
    for d, m in (("knu", mk), ("mimic", mmk)):
        s, Y, pat = C[d]["s"][m], C[d]["Y"][m, h], C[d]["pat"][m]; b = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
        x, y, lo, hi = [], [], [], []
        for i in range(10):
            sel = b == i
            if sel.sum() < 20: continue
            x.append(np.median(s[sel])); y.append(100 * Y[sel].mean()); l, u = pboot(Y[sel], pat[sel]); lo.append(100 * l); hi.append(100 * u)
        y, lo, hi = map(np.array, (y, lo, hi))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], linestyle="-", capsize=1.5, elinewidth=fs.LW_CHROME, **fs.series_kw(d, linewidth=1.1, markersize=4.5))
    fs.clean_axes(ax, "y"); ax.set_xlabel("Shared readiness score $s(z)$", fontsize=8.5); ax.tick_params(labelsize=7.5)
    ax.set_title(title if title else f"Switch within {h+1} day{'s' if h else ''}", loc="left", fontsize=8.5)

def fig2(C):
    """Figure 2: A observed switching by score decile, B absolute risk difference."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), sharex="row", sharey="row")
    for h, ax in enumerate(axes[0]): decile_panel(ax, C, h)
    axes[0, 0].set_ylabel("Observed switching (%)", fontsize=8.5); axes[0, 0].set_ylim(0, 100); axes[0, 0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0, 0].legend(handles=fs.legend_handles(), frameon=False, loc="upper left", fontsize=8)
    rd_row(axes[1], C, legend=True)
    for ax, lab in ((axes[0, 0], "A"), (axes[1, 0], "B")):
        ax.text(-0.28, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(h_pad=2.2, w_pad=1.0); fs.save(fig, OUT / "figure2.pdf")

def fig4():
    """Forest of cohort log-ORs per sensitivity condition, at two evaluation points."""
    log = open(REPO / "logs" / "cv_logor_matrix.log").read()
    blocks = {"median": log.split("pooled median")[1].split("pooled p90")[0], "p90": log.split("pooled p90")[1]}
    order = ["All anchors", "First 7 anchor-days", "Anchor-position reweighted", "ICU", "Non-ICU", "Sepsis-3", "Bacteraemia"]
    disp = {"All anchors": "All", "First 7 anchor-days": "Anchor-days\n1–7 only", "Anchor-position reweighted": "Anchor-position\nmatched", "Sepsis-3": "Sepsis-3", "Bacteraemia": "Positive\nblood culture"}
    HCOL = {0: "#9ECAE1", 1: "#3182BD", 2: "#08306B"}; off = {0: -0.22, 1: 0.0, 2: 0.22}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (pt, txt) in zip(axes, blocks.items()):
        rows = {m.group(1).strip(): [(float(a), float(b), float(c)) for a, b, c in re.findall(r"([+-]?[\d.]+) \(\s*([+-]?[\d.]+)-\s*([+-]?[\d.]+)\)", m.group(2))]
                for m in re.finditer(r"^\s+(.+?)\s{2,}((?:[+-]?[\d.]+ \(\s*[+-]?[\d.]+-\s*[+-]?[\d.]+\)\s*){3})$", txt, re.M)}
        y = np.arange(len(order))[::-1]
        for h in range(3):
            est = np.array([rows[o][h][0] for o in order]); lo = np.array([rows[o][h][1] for o in order]); hi = np.array([rows[o][h][2] for o in order])
            ax.errorbar(est, y + off[h], xerr=[est - lo, hi - est], fmt="o", color=HCOL[h], markersize=3.8, capsize=1.8, elinewidth=1.0, markeredgecolor=fs.SURFACE, markeredgewidth=0.6, label=f"{h+1}-day")
        ax.set_yticks(y); ax.set_yticklabels([disp.get(o, o) for o in order], fontsize=8); ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlim(left=-0.05); ax.grid(True, axis="x", color=fs.GRID, lw=fs.LW_CHROME, zorder=0); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Cohort log-odds ratio, MIMIC-IV vs KNU", fontsize=8.5)
        ax.set_title("At the median shared score" if pt == "median" else "At the 90th percentile shared score", loc="center", fontsize=8.5)
    axes[1].legend(title="Horizon", frameon=False, loc="lower right", fontsize=7.5, title_fontsize=7.5)
    fig.tight_layout(w_pad=1.0); fs.save(fig, OUT / "figure3.pdf")

# Figure 5: A not-switched vs score percentile, B inpatient days after the switch
def fig5(C):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6))
    ax = axes[0]; pooled = np.r_[C["knu"]["s"], C["mimic"]["s"]]; q = np.quantile(pooled, np.linspace(0, 1, 21))
    for d in DOMAINS:
        s, Y3 = C[d]["s"], C[d]["Y"][:, 2]; b = np.clip(np.digitize(s, q[1:-1]), 0, 19)
        x = [(i + 0.5) * 5 for i in range(20)]; y = [100 * (1 - Y3[b == i].mean()) if (b == i).sum() >= 20 else np.nan for i in range(20)]
        ax.plot(x, y, **fs.series_kw(d))
    fs.clean_axes(ax, "y"); ax.set_xlabel("Shared readiness percentile (pooled)", fontsize=6.5); ax.set_ylabel("Not switched within 3 days (%)", fontsize=6.5); ax.tick_params(labelsize=5.5)
    ax = axes[1]; cats = [("≤0", lambda v: v <= 0), ("1", lambda v: v == 1), ("2–3", lambda v: (v >= 2) & (v <= 3)), ("4–7", lambda v: (v >= 4) & (v <= 7)), (">7", lambda v: v > 7)]
    width = 0.38
    for j, d in enumerate(DOMAINS):
        ep = pd.read_parquet(RE / f"episode_table_{d}.parquet"); v = (pd.to_datetime(ep.followup_end) - pd.to_datetime(ep.switch_terminal)).dt.days.to_numpy()
        vals = [100 * f(v).mean() for _, f in cats]; ax.bar(np.arange(5) + (j - 0.5) * width, vals, width, color=fs.COLOR[d], label=fs.LABEL[d], edgecolor=fs.SURFACE, linewidth=0.6)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels([c for c, _ in cats]); fs.clean_axes(ax, "y"); ax.set_xlabel("Inpatient days after the switch", fontsize=6.5); ax.set_ylabel("Episodes (%)", fontsize=6.5); ax.tick_params(labelsize=5.5); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(w_pad=1.2); fs.save(fig, OUT / "figure5.pdf")

# Reliability curves
def fig6(C):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for i, d in enumerate(DOMAINS):
        for h in range(3):
            ax = axes[i, h]; p, y = C[d]["p"][:, h], C[d]["Y"][:, h]; e = np.linspace(0, 1, 11); b = np.clip(np.digitize(p, e[1:-1]), 0, 9)
            x = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 30]; o = [y[b == k].mean() for k in range(10) if (b == k).sum() >= 30]
            ax.plot([0, 1], [0, 1], color=fs.INK_MUTED, lw=fs.LW_CHROME, ls="--"); ax.plot(x, o, **fs.series_kw(d))
            brier = np.mean((p - y) ** 2); ece = sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any())
            ax.text(0.04, 0.92, f"Brier {brier:.3f}\nECE {ece:.3f}", transform=ax.transAxes, va="top", fontsize=7, color=fs.INK_SECONDARY)
            fs.clean_axes(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 0: ax.set_title(f"Switch within {h+1} day{'s' if h else ''}", loc="center", fontsize=8.5)
            if i == 1: ax.set_xlabel("Predicted probability")
            if h == 0: ax.set_ylabel(f"{fs.LABEL[d]}\nObserved proportion")
    fig.tight_layout(w_pad=0.8, h_pad=0.8); fs.save(fig, OUT / "Sup" / "FigureS_calibration.pdf")

# Figure S1: ICU strata
def figS1(C):
    icu = icu_flags(C)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    decile_panel(axes[0], C, 0, mask={d: ~icu[d] for d in DOMAINS}, title="Non-ICU anchor-days")
    decile_panel(axes[1], C, 0, mask={d: icu[d] for d in DOMAINS}, title="ICU anchor-days")
    axes[0].legend(handles=fs.legend_handles(), frameon=False, loc="upper left"); fig.tight_layout(w_pad=1.2); fs.save(fig, OUT / "Sup" / "FigureS1.pdf")


# Figure 1: schematic
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    def box(x, y, w, h, text, fc="#F4F4F2", ec=fs.INK_SECONDARY, fs_=6.9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, color=fs.INK, fontweight="bold" if bold else "normal", linespacing=1.3)
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=fs.INK_SECONDARY, shrinkA=0, shrinkB=0))
    # cohorts
    box(1, 27, 17, 12, "KNU\n9,503 episodes\n148,361 anchor-days", fc="#F7E6E8", ec=fs.KNU)
    box(1, 5, 17, 12, "MIMIC-IV\n8,712 episodes\n60,242 anchor-days", fc="#E4EEF8", ec=fs.MIMIC)
    # representation
    box(23, 13, 20, 18, "72-h window\nbefore each anchor day\n(time, channel, value)\ntriplets, ≤384 tokens\nno imputation")
    # encoder
    box(48, 13, 21, 18, "Shared encoder\nSTraTS Transformer\n64-d, 2 layers, 4 heads\nattention pooling\nconditioned on age, sex")
    # outputs
    box(73, 27, 26.5, 12, "Shared readiness score s(z)\none axis for both cohorts", fc="#EFEFEC", bold=True, fs_=6.6)
    box(73, 5, 26.5, 12, "Cohort cutpoints c[d, h]\nlogit = 3·(s − c[d, h])\nh = 1, 2, 3 days")
    arrow(18, 33, 23, 25); arrow(18, 11, 23, 19); arrow(43, 22, 48, 22); arrow(69, 24, 73, 31); arrow(69, 20, 73, 13)
    ax.text(58.5, 9.5, "MMD aligns the two cohorts'\npooled representations", ha="center", va="top", fontsize=6.3, color=fs.INK_MUTED, linespacing=1.2)
    ax.text(86.5, 1.0, "gap = c50(KNU) − c50(MIMIC-IV)\nat true switch probability 50%", ha="center", va="bottom", fontsize=6.3, color=fs.INK_SECONDARY, linespacing=1.2)
    fs.save(fig, OUT / "figure1.pdf")


def rd_row(axes, C, legend=True):
    """MIMIC-IV minus KNU absolute risk difference across the shared score.
    logit(y) = b0 + b1 s + b2 d + b3 s d, patient cluster-robust, delta-method CI."""
    from scipy.special import expit
    pooled = np.r_[C["knu"]["s"], C["mimic"]["s"]]; lo, hi = np.percentile(pooled, [2.5, 97.5]); grid = np.linspace(lo, hi, 300)
    p10, p50, p90 = np.percentile(pooled, [10, 50, 90])
    print("  Figure S (risk difference) — MIMIC-IV vs KNU, held-out test, 15-model mean score")
    ymax = 0
    for h in range(3):
        y = np.r_[C["knu"]["Y"][:, h], C["mimic"]["Y"][:, h]]; d = np.r_[np.zeros(len(C["knu"]["s"])), np.ones(len(C["mimic"]["s"]))]
        g = np.r_[C["knu"]["pat"], C["mimic"]["pat"]]
        X = sm.add_constant(np.column_stack([pooled, d, pooled * d]))
        r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g}); b, V = r.params, r.cov_params()
        def rd(sv):
            pk = expit(b[0] + b[1] * sv); pm = expit(b[0] + b[1] * sv + b[2] + b[3] * sv)
            gk = pk * (1 - pk); gm = pm * (1 - pm)
            J = np.column_stack([gm - gk, (gm - gk) * sv, gm, gm * sv]); se = np.sqrt(np.einsum("ij,jk,ik->i", J, V, J))
            return pm - pk, se
        est, se = rd(grid); ax = axes[h]
        ax.fill_between(grid, 100 * (est - 1.96 * se), 100 * (est + 1.96 * se), color=fs.MIMIC, alpha=0.18, lw=0)
        inner = (grid >= p10) & (grid <= p90)
        ax.plot(grid[inner], 100 * est[inner], color=fs.INK, lw=fs.LW_DATA)
        ax.plot(grid[grid < p10], 100 * est[grid < p10], color=fs.INK, lw=fs.LW_DATA, ls="--"); ax.plot(grid[grid > p90], 100 * est[grid > p90], color=fs.INK, lw=fs.LW_DATA, ls="--")
        ax.axhline(0, color=fs.INK_MUTED, lw=fs.LW_CHROME, ls="--")
        for q, lab in ((p10, "P10"), (p90, "P90")):
            ax.axvline(q, color=fs.INK_MUTED, lw=fs.LW_CHROME, ls=":"); ax.text(q, 0.97, lab, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.5, color=fs.INK_MUTED)
        for q, dx, ha in ((p10, 0.04, "left"), (p90, -0.04, "right")):
            e, _ = rd(np.array([q])); ax.plot([q], [100 * e[0]], "o", color=fs.KNU, markersize=4, markeredgecolor=fs.SURFACE, markeredgewidth=0.8, zorder=5)
            ax.annotate(f"{100*e[0]:.1f} pp", xy=(q, 100 * e[0]), xytext=(q + dx, 100 * e[0] + 3), fontsize=7, ha=ha, color=fs.INK)
        ymax = max(ymax, float(100 * (est + 1.96 * se).max()))
        fs.clean_axes(ax, "y"); ax.set_title(f"{h+1}-day horizon", loc="center", fontsize=8.5); ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Shared readiness score $s(z)$", fontsize=8.5)
        e10, s10 = rd(np.array([p10])); e90, s90 = rd(np.array([p90])); e50, s50 = rd(np.array([p50]))
        print(f"    h{h+1}: OR at s=0 {np.exp(b[2]):.2f}; RD p10 {100*e10[0]:.1f} ({100*(e10[0]-1.96*s10[0]):.1f}–{100*(e10[0]+1.96*s10[0]):.1f}) pp, p50 {100*e50[0]:.1f}, p90 {100*e90[0]:.1f} ({100*(e90[0]-1.96*s90[0]):.1f}–{100*(e90[0]+1.96*s90[0]):.1f}) pp; interaction p={r.pvalues[3]:.3f}")
    top = np.ceil(ymax / 10) * 10 + 5
    for ax in axes:
        ax.set_ylim(-top * 0.12, top)
        for dom in DOMAINS:  # density strip in the bottom 12% of the axes
            hist, edges = np.histogram(C[dom]["s"], bins=np.linspace(lo, hi, 60), density=True); hist = hist / hist.max() * top * 0.11
            ax.fill_between((edges[:-1] + edges[1:]) / 2, -top * 0.12, -top * 0.12 + hist, color=fs.COLOR[dom], alpha=0.30, lw=0, label=fs.LABEL[dom])
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
    axes[0].set_ylabel("MIMIC-IV − KNU\nswitch probability (pp)", fontsize=8.5)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=fs.COLOR[dm], alpha=0.30) for dm in DOMAINS]
        axes[0].legend(handles, [fs.LABEL[dm] for dm in DOMAINS], frameon=False, loc="upper right", fontsize=7.5)

def fig3(C):
    """Standalone version of the Figure 2B risk-difference panels."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    rd_row(axes, C); fig.tight_layout(w_pad=1.0); fs.save(fig, OUT / "Sup" / "FigureS_riskdiff.pdf")



if __name__ == "__main__":
    want = sys.argv[1:] or ["cache", "1", "2", "3", "4", "5", "6", "S1"]
    if "cache" in want or not CACHE.exists(): build_cache()
    C = load_cache()
    if "1" in want: fig1()
    if "2" in want: fig2(C)
    if "3" in want: fig3(C)
    if "4" in want: fig4()
    if "5" in want: fig5(C)
    if "6" in want: fig6(C)
    if "S1" in want: figS1(C)
