"""c50 gap per HPO candidate, used as the final selection criterion once discrimination is
tied. Prints seed mean +/- SD per trial.
"""
import os
import glob
import re

import numpy as np

from common import DOMAINS, c50, load_ckpt, rule, score_split

SC = os.environ.get("IVOS_SCRATCH", "/tmp/inertia_scratch")
rows = {}
for p in sorted(glob.glob(f"{SC}/out_cf_t*/inertia_net/*/inertia_net_model.pth")):
    m = re.search(r"cf_t(\d+)_s(\d+)", p)
    t = int(m.group(1))
    enc, hd, ck = load_ckpt(p, "cuda:0")
    S, Y = {}, {}
    for dom in DOMAINS:
        r = score_split(None, dom, "test", "cuda:0", enc, hd, ck)
        S[dom], Y[dom] = r["s"], r["Y"]
    g = [c50(S["knu"], Y["knu"][:, h]) - c50(S["mimic"], Y["mimic"][:, h]) for h in range(3)]
    rows.setdefault(t, []).append(g + [ck["hp"].get("lambda_mmd", np.nan)])
    print(f"  t{t} s{m.group(2)}  gap h1 {g[0]:.3f}  h2 {g[1]:.3f}  h3 {g[2]:.3f}", flush=True)

rule("c50 gap - seed mean +/- SD per candidate")
print(f"  {'trial':>6s} {'MMD':>5s} {'h1':>16s} {'h2':>16s} {'h3':>16s}")
for t, v in sorted(rows.items()):
    a = np.array(v, float)
    print(f"  t{t:<5d} {a[0,3]:>5.1f} " +
          " ".join(f"{a[:,h].mean():>8.3f} ±{a[:,h].std():.3f}" for h in range(3)) +
          ("   <- current spec" if t == 0 else ""))
