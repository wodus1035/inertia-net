"""MIMIC blood-culture organism mix, classified the same way as KNU, restricted to
BLOOD CULTURE specimens; prints the side-by-side table to stdout.

Organism and Specimen_Type are parallel pipe-joined lists, so tokens are matched by
position rather than by substring search over the whole field.
"""
import pandas as pd
from common import DATA
from src.config import MIMIC_SRC, KNU_SRC

COMMENSAL = ("COAGULASE NEGATIVE", "EPIDERMIDIS", "CAPITIS", "HOMINIS", "HAEMOLYTICUS",
             "WARNERI", "SAPROPHYTICUS", "CORYNEBACTERIUM", "PROPIONIBACTERIUM",
             "CUTIBACTERIUM", "BACILLUS", "MICROCOCCUS", "VIRIDANS", "AEROCOCCUS",
             "RHODOCOCCUS", "DIPHTHEROIDS")
def commensal(n):
    s = str(n).upper()
    return ("LUGDUNENSIS" not in s) and any(c in s for c in COMMENSAL)

def cls(n):
    """Shared classifier for both cohorts; MIMIC also spells it 'STAPH AUREUS COAG +'."""
    s = str(n).upper()
    if "MRSA" in s or "METHICILLIN RESISTANT STAPH" in s: return "S. aureus"
    if "STAPH AUREUS" in s or "STAPHYLOCOCCUS AUREUS" in s or "LUGDUNENSIS" in s: return "S. aureus"
    if "CANDIDA" in s or "YEAST" in s or "CRYPTOCOC" in s or "ASPERGILL" in s: return "Candida/fungi"
    if "ENTEROCOCCUS" in s: return "Enterococcus"
    if "PSEUDOMONAS" in s: return "Pseudomonas"
    if "ACINETOBACTER" in s: return "Acinetobacter"
    if any(x in s for x in ("ESCHERICHIA", "E. COLI", "KLEBSIELLA", "PROTEUS", "ENTEROBACTER",
                            "SERRATIA", "CITROBACTER", "MORGANELLA")): return "Enteric GNR"
    if "STREPTOCOC" in s and "VIRIDANS" not in s: return "Streptococcus"
    if commensal(n): return "Commensal (CoNS etc.)"
    return "Other"

def label(sset):
    s2 = sset - {"Commensal (CoNS etc.)"}
    if not s2: return "Commensal only"
    if len(s2) > 1: return "Polymicrobial"
    return list(s2)[0]

m = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "Organism", "Specimen_Type", "switch_flag"])
m = m[m.switch_flag == 1].dropna(subset=["Organism", "Specimen_Type"])
m["Episode_ID"] = m.Episode_ID.astype(str)
sp = m.Specimen_Type.astype(str).str.split(r"\s*\|\s*")
og = m.Organism.astype(str).str.split(r"\s*\|\s*")
ok = sp.str.len() == og.str.len()
print(f"Organism/Specimen_Type token counts agree in {ok.mean():.1%} of rows")
m, sp, og = m[ok], sp[ok], og[ok]
ex = pd.DataFrame({"Episode_ID": m.Episode_ID.values, "sp": sp.values, "og": og.values}).explode(["sp", "og"])
ex["sp"] = ex.sp.str.strip().str.upper(); ex["og"] = ex.og.str.strip()
ex = ex[ex.og.ne("") & ex.og.str.upper().ne("NAN")]
blood = ex[ex.sp.str.contains("BLOOD CULTURE", na=False)]
print(f"\nepisodes with any positive specimen {ex.Episode_ID.nunique():,}"
      f"   blood-culture positive {blood.Episode_ID.nunique():,}")

coh_m = pd.read_parquet(f"{DATA}/mimic_meta_d3_b36x2h.parquet",
                        columns=["Episode_ID"]).Episode_ID.astype(str).unique()
print(f"of {len(coh_m):,} MIMIC paper-cohort episodes")
print(f"   blood-culture positive : {len(set(blood.Episode_ID) & set(coh_m)):,} "
      f"({len(set(blood.Episode_ID) & set(coh_m))/len(coh_m):.1%})")
print(f"   any specimen positive  : {len(set(ex.Episode_ID) & set(coh_m)):,} "
      f"({len(set(ex.Episode_ID) & set(coh_m))/len(coh_m):.1%})")

blood = blood[blood.Episode_ID.isin(coh_m)]
blood["cl"] = blood.og.map(cls)
mg = blood.groupby("Episode_ID").cl.apply(set).map(label)

# KNU side, same classifier
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Organism"]).dropna(subset=["Organism"])
k["Episode_ID"] = k.Episode_ID.astype(str)
coh_k = pd.read_parquet(f"{DATA}/knu_meta_d3_b36x2h.parquet",
                        columns=["Episode_ID"]).Episode_ID.astype(str).unique()
k = k[k.Episode_ID.isin(coh_k)]
ko = k.assign(o=k.Organism.astype(str).str.split(r"\s*\|\s*")).explode("o")
ko["o"] = ko.o.str.strip(); ko = ko[ko.o.ne("")]
ko["cl"] = ko.o.map(cls)
kg = ko.groupby("Episode_ID").cl.apply(set).map(label)

print(f"\n{'='*74}\nblood-culture organism mix, same classifier, paper cohorts\n{'='*74}")
a = kg.value_counts(normalize=True).mul(100)
b = mg.value_counts(normalize=True).mul(100)
t = pd.DataFrame({f"KNU (n={len(kg):,})": a, f"MIMIC blood (n={len(mg):,})": b}).fillna(0)
t["diff"] = t.iloc[:, 0] - t.iloc[:, 1]
print(t.round(1).sort_values(t.columns[0], ascending=False).to_string())

print(f"\ntop 15 raw organism names in blood cultures (MIMIC, paper cohort)")
print(blood.og.value_counts().head(15).to_string())
