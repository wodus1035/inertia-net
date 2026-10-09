"""Organism mix of KNU bacteremia episodes and its relation to IV duration; compares
the same classifier against MIMIC. Prints tables to stdout only.
"""
import pandas as pd
import common  # path bootstrap
from src.config import KNU_SRC, MIMIC_SRC
from common import DATA


COMMENSAL = ("COAGULASE NEGATIVE", "EPIDERMIDIS", "CAPITIS", "HOMINIS", "HAEMOLYTICUS",
             "WARNERI", "SAPROPHYTICUS", "CORYNEBACTERIUM", "PROPIONIBACTERIUM",
             "CUTIBACTERIUM", "BACILLUS", "MICROCOCCUS", "VIRIDANS", "AEROCOCCUS",
             "RHODOCOCCUS", "DIPHTHEROIDS")
def commensal(n):
    s = str(n).upper()
    return ("LUGDUNENSIS" not in s) and any(c in s for c in COMMENSAL)

# groups for which prolonged IV is guideline standard (S. aureus and candidemia >=14d,
# enterococcal endocarditis 4-6 weeks) vs enteric gram-negatives at 7-14d
def cls(n):
    s = str(n).upper()
    if "STAPHYLOCOCCUS AUREUS" in s or "LUGDUNENSIS" in s: return "S. aureus"
    if "CANDIDA" in s or "CRYPTOCOC" in s: return "Candida/fungi"
    if "ENTEROCOCCUS" in s: return "Enterococcus"
    if "PSEUDOMONAS" in s: return "Pseudomonas"
    if "ACINETOBACTER" in s: return "Acinetobacter"
    if any(x in s for x in ("ESCHERICHIA", "KLEBSIELLA", "PROTEUS", "ENTEROBACTER",
                            "SERRATIA", "CITROBACTER", "MORGANELLA")): return "Enteric GNR"
    if "STREPTOCOCCUS" in s: return "Streptococcus"
    if commensal(n): return "Commensal (CoNS etc.)"
    return "Other"

k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Event_Date", "Drug_Codes", "Organism"])
k = k.dropna(subset=["Episode_ID", "Event_Date"]); k["Episode_ID"] = k.Episode_ID.astype(str)
k["d"] = pd.to_datetime(k.Event_Date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
k = k.dropna(subset=["d"])

# organism-class set per episode
org = (k[["Episode_ID", "Organism"]].dropna().assign(
        o=lambda x: x.Organism.astype(str).str.split(r"\s*\|\s*"))
       .explode("o"))
org["o"] = org.o.str.strip()
org = org[org.o.ne("")]
org["cl"] = org.o.map(cls)
epi_org = org.groupby("Episode_ID").cl.apply(set)

# IV duration, Bolton style: first IV day to last IV day
def has(c, p): return isinstance(c, str) and any(x.strip().upper().startswith(p)
                                                 for x in c.split("|") if x.strip())
k["iv"] = k.Drug_Codes.map(lambda c: has(c, "DI")); k["po"] = k.Drug_Codes.map(lambda c: has(c, "DO"))
g = k.groupby(["Episode_ID", "d"]).agg(iv=("iv", "any"), po=("po", "any")).reset_index()
a = g[g.iv].groupby("Episode_ID").d.agg(iv_start="min", iv_stop="max")
b = g[g.po].groupby("Episode_ID").d.agg(po_start="min", po_stop="max")
w = a.join(b, how="inner").reset_index()
w["iv_dur"] = (w.iv_stop - w.iv_start).dt.days
w = w[(w.iv_stop <= w.po_stop) & (w.iv_dur >= 0)]

coh = set(pd.read_parquet(f"{DATA}/knu_meta_d3_b36x2h.parquet",
                          columns=["Episode_ID"]).Episode_ID.astype(str))
w["in_cohort"] = w.Episode_ID.isin(coh)
w["cls"] = w.Episode_ID.map(epi_org)
print(f"KNU: episodes with both IV and oral {len(w):,}  (in paper cohort {w.in_cohort.sum():,})")
print(f"  episodes with organism info {w.cls.notna().sum():,} ({w.cls.notna().mean():.1%})")

cw = w[w.in_cohort & w.cls.notna()].copy()
print(f"\n{'='*80}\nKNU paper cohort {len(cw):,} episodes - organism mix and IV duration\n{'='*80}")

# an episode with more than one non-commensal class is polymicrobial
def label(s):
    s2 = s - {"Commensal (CoNS etc.)"}
    if not s2: return "Commensal only"
    if len(s2) > 1: return "Polymicrobial"
    return list(s2)[0]
cw["grp"] = cw.cls.map(label)
rows = []
for gname, sub in cw.groupby("grp"):
    rows.append((gname, len(sub), len(sub)/len(cw), sub.iv_dur.median(),
                 sub.iv_dur.mean(), (sub.iv_dur >= 14).mean(), (sub.iv_dur <= 7).mean()))
t = pd.DataFrame(rows, columns=["organism", "n", "share", "iv_med", "iv_mean", ">=14d", "<=7d"]
                 ).sort_values("n", ascending=False)
print(f"  {'organism':>16s} {'n':>6s} {'share':>7s} {'iv_med':>7s} {'iv_mean':>7s} {'>=14d':>8s} {'<=7d':>7s}")
for _, r in t.iterrows():
    print(f"  {r['organism']:>16s} {r['n']:>6,} {r['share']:>7.1%} {r['iv_med']:>7.0f} "
          f"{r['iv_mean']:>7.1f} {r['>=14d']:>8.1%} {r['<=7d']:>7.1%}")
print(f"  {'total':>16s} {len(cw):>6,} {1:>7.1%} {cw.iv_dur.median():>7.0f} "
      f"{cw.iv_dur.mean():>7.1f} {(cw.iv_dur>=14).mean():>8.1%} {(cw.iv_dur<=7).mean():>7.1%}")

print(f"\n  episodes containing a commensal: {cw.cls.map(lambda s: 'Commensal (CoNS etc.)' in s).mean():.1%}")
print(f"  episodes that are commensal-only: {(cw.grp=='Commensal only').mean():.1%}"
      f"   -> the share KNU would lose under the MIMIC bacteremia definition")

# MIMIC comparison
m = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "Organism", "switch_flag"]).dropna(subset=["Organism"])
m = m[m.switch_flag == 1]
mo = m.assign(o=lambda x: x.Organism.astype(str).str.split(r"\s*\|\s*")).explode("o")
mo["o"] = mo.o.str.strip(); mo = mo[mo.o.ne("")]
mo["cl"] = mo.o.map(cls)
me = mo.groupby("Episode_ID").cl.apply(set).map(label)
print(f"\n{'='*80}\nMIMIC paper cohort, episodes with an organism {me.nunique() if False else len(me):,}\n{'='*80}")
print(me.value_counts(normalize=True).mul(100).round(1).to_string())


# staphylococcal species detail: the CoNS bucket mixes S. lugdunensis (as virulent as
# S. aureus) with the common catheter contaminants
print(f"\n{'='*80}\nKNU staphylococci by species (per episode, episodes counted more than once)\n{'='*80}")
staph = org[org.o.str.upper().str.contains("STAPHYLOCOC", na=False)].copy()
staph["sp"] = staph.o.str.strip()
epi_staph = staph.groupby(["Episode_ID", "sp"]).size().reset_index(name="n")
dur = w.set_index("Episode_ID").iv_dur
epi_staph["iv_dur"] = epi_staph.Episode_ID.map(dur)
epi_staph["in_coh"] = epi_staph.Episode_ID.isin(coh)
q = epi_staph[epi_staph.in_coh & epi_staph.iv_dur.notna()]
tot_ep = q.Episode_ID.nunique()
print(f"  episodes in cohort with a staphylococcus {tot_ep:,}")
print(f"  {'species':>42s} {'episodes':>8s} {'share':>6s} {'iv_med':>7s} {'>=14d':>7s}")
for sp, sub in sorted(q.groupby("sp"), key=lambda x: -x[1].Episode_ID.nunique())[:14]:
    ne = sub.Episode_ID.nunique()
    if ne < 20: continue
    print(f"  {sp[:42]:>42s} {ne:>8,} {ne/tot_ep:>6.1%} {sub.iv_dur.median():>7.0f} "
          f"{(sub.iv_dur>=14).mean():>7.1%}")

print(f"\n  under the NHSN commensal rule:")
sps = sorted(q.sp.unique())
com = [x for x in sps if commensal(x)]
non = [x for x in sps if not commensal(x)]
print(f"    species classed as commensal: {len(com)} - e.g. {', '.join(com[:6])}")
print(f"    species kept as pathogens:   {len(non)} - e.g. {', '.join(non[:6])}")
