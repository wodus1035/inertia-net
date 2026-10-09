"""Table S5: observed switch rate by pooled decile of s, three horizons, held-out test.
Scores are the 15-model mean; CIs from 1,000 patient-cluster bootstraps.
Writes the LaTeX table rows to stdout."""
import numpy as np
from common import CV_TAGS, DOMAINS, rule, score_models
S, D = score_models(CV_TAGS, DOMAINS, split="test", verbose=False)
s = {d: np.mean(S[d], 0) for d in DOMAINS}; Y = {d: D[d]["Y"] for d in DOMAINS}; P = {d: np.asarray(D[d]["pat"]) for d in DOMAINS}
edges = np.quantile(np.r_[s["knu"], s["mimic"]], np.linspace(0, 1, 11)); lo_e, hi_e = edges.copy(), edges.copy()
dec = {d: np.clip(np.digitize(s[d], edges[1:-1]), 0, 9) for d in DOMAINS}
rng = np.random.RandomState(0); NB = 1000
boot = {}
for d in DOMAINS:
    uq, inv = np.unique(P[d], return_inverse=True); where = [np.where(inv == k)[0] for k in range(len(uq))]
    acc = np.full((NB, 10, 3), np.nan)
    for b in range(NB):
        idx = np.concatenate([where[k] for k in rng.randint(0, len(uq), len(uq))])
        dd, yy = dec[d][idx], Y[d][idx]
        for i in range(10):
            m = dd == i
            if m.any(): acc[b, i] = yy[m].mean(0)
    boot[d] = acc
rule("Table S5 - LaTeX rows (15-model mean score, 1,000 patient bootstraps)")
fmt = lambda x: f"{x:+.3f}" if np.isfinite(x) else ""
for h, lab in enumerate(("Switch within 1 day", "Switch within 2 days", "Switch within 3 days")):
    print(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}} \\\\")
    for i in range(10):
        lo = f"${edges[i]:+.3f}$" if i > 0 else f"${np.r_[s['knu'], s['mimic']].min():+.3f}$"
        hi = f"${edges[i+1]:+.3f}$" if i < 9 else f"${np.r_[s['knu'], s['mimic']].max():+.3f}$"
        cells = []
        for d in DOMAINS:
            m = dec[d] == i; p = 100 * Y[d][m, h].mean(); ci = 100 * np.nanpercentile(boot[d][:, i, h], [2.5, 97.5])
            cells.append(f"{p:.1f} ({ci[0]:.1f}--{ci[1]:.1f}) & {m.sum():,}")
        print(f"{i+1} & {lo} to {hi} & {cells[0]} & {cells[1]} \\\\")
    print("\\hline")
nk = len(set(P["knu"])); nm = len(set(P["mimic"])); print(f"\nfor the footnote: {nk + nm:,} unique patients across the two cohorts (KNU {nk:,}, MIMIC-IV {nm:,})")
