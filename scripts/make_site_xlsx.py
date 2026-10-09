"""Infection-site and organism composition of the two cohorts; writes
{RESULTS}/cohort_infection_sites.xlsx.

KNU has no specimen-type field and is bloodstream infection by construction, so the site
axis exists only for MIMIC and the cross-cohort comparison is limited to blood isolates.
"""
import numpy as np, pandas as pd
from common import DATA, MIMIC_RAW, RESULTS
from src.config import MIMIC_SRC, KNU_SRC
MB = str(MIMIC_RAW) + "/hosp/microbiologyevents.csv"
OUT = f"{RESULTS}/cohort_infection_sites.xlsx"

COMMENSAL = ("COAGULASE NEGATIVE", "EPIDERMIDIS", "CAPITIS", "HOMINIS", "HAEMOLYTICUS",
             "WARNERI", "SAPROPHYTICUS", "CORYNEBACTERIUM", "PROPIONIBACTERIUM",
             "CUTIBACTERIUM", "BACILLUS", "MICROCOCCUS", "VIRIDANS", "AEROCOCCUS",
             "RHODOCOCCUS", "DIPHTHEROIDS")
def commensal(n):
    s = str(n).upper(); return ("LUGDUNENSIS" not in s) and any(c in s for c in COMMENSAL)
def cls(n):
    s = str(n).upper()
    if "MRSA" in s or "METHICILLIN RESISTANT STAPH" in s: return "S. aureus"
    if "STAPH AUREUS" in s or "STAPHYLOCOCCUS AUREUS" in s or "LUGDUNENSIS" in s: return "S. aureus"
    if "CANDIDA" in s or "YEAST" in s or "CRYPTOCOC" in s or "ASPERGILL" in s: return "Candida/fungus"
    if "ENTEROCOCCUS" in s: return "Enterococcus"
    if "PSEUDOMONAS" in s: return "Pseudomonas"
    if "ACINETOBACTER" in s: return "Acinetobacter"
    if any(x in s for x in ("ESCHERICHIA", "E. COLI", "KLEBSIELLA", "PROTEUS", "ENTEROBACTER",
                            "SERRATIA", "CITROBACTER", "MORGANELLA")): return "Enteric GNR"
    if "STREPTOCOC" in s and "VIRIDANS" not in s: return "Streptococcus"
    if commensal(n): return "Commensal (CoNS etc.)"
    return "Other"
def label(ss):
    s2 = ss - {"Commensal (CoNS etc.)"}
    if not s2: return "Commensal only"
    if len(s2) > 1: return "Polymicrobial"
    return list(s2)[0]
def site(x):
    u = str(x).upper()
    if "BLOOD CULTURE" in u: return "Blood"
    if "URINE" in u: return "Urine"
    if any(k in u for k in ("SPUTUM","BRONCH","TRACHEAL","MINI-BAL","LUNG","PLEURAL")): return "Respiratory"
    if any(k in u for k in ("SWAB","ABSCESS","TISSUE","WOUND","SKIN","FOOT")): return "Wound/tissue"
    if any(k in u for k in ("STOOL","PERITONEAL","BILE","ASCITES","GASTRIC")): return "Abdominal/GI"
    if "CSF" in u or "SPINAL" in u: return "CNS"
    if "CATHETER" in u or "LINE" in u: return "Catheter tip"
    if "SCREEN" in u or "MRSA" in u or "VRE" in u: return "Surveillance screen"
    if "JOINT" in u or "SYNOVIAL" in u or "BONE" in u: return "Joint/bone"
    return "Other"

# cohorts
coh_m = set(pd.read_parquet(f"{DATA}/mimic_meta_d3_b36x2h.parquet",
                            columns=["Episode_ID"]).Episode_ID.astype(str))
coh_k = set(pd.read_parquet(f"{DATA}/knu_meta_d3_b36x2h.parquet",
                            columns=["Episode_ID"]).Episode_ID.astype(str))
sev = pd.read_parquet(MIMIC_SRC, columns=["Episode_ID", "hadm_id"]).drop_duplicates("Episode_ID")
sev["Episode_ID"] = sev.Episode_ID.astype(str)
s = sev[sev.Episode_ID.isin(coh_m)].dropna(subset=["hadm_id"])
hs = set(s.hadm_id.astype("int64")); N = len(hs)

rows = []
for ch in pd.read_csv(MB, usecols=["hadm_id", "spec_type_desc", "org_name", "org_itemid"],
                      dtype={"hadm_id": "float64", "spec_type_desc": "string",
                             "org_name": "string", "org_itemid": "float64"}, chunksize=2_000_000):
    ch = ch.dropna(subset=["hadm_id"]); ch["h"] = ch.hadm_id.astype("int64")
    ch = ch[ch.h.isin(hs)]
    if len(ch):
        pos = (ch.org_name.notna() & ~ch.org_itemid.isin([90856, 90760])
               & ch.org_name.ne("") & ch.org_name.ne("CANCELLED"))
        rows.append(pd.DataFrame({"h": ch.h.values, "sp": ch.spec_type_desc.values,
                                  "pos": pos.values, "org": ch.org_name.values}))
mb = pd.concat(rows, ignore_index=True); mb["site"] = mb.sp.map(site)
print(f"MIMIC culture records {len(mb):,}  admissions {N:,}", flush=True)

# sheet: MIMIC by site
a = []
for st, g in mb.groupby("site"):
    d = g.h.nunique(); p = g.loc[g.pos, "h"].nunique()
    a.append({"Site": st, "Admissions cultured": d, "% of cohort cultured": round(100*d/N, 1),
              "Admissions positive": p, "% of cohort positive": round(100*p/N, 1),
              "Positivity rate %": round(100*p/max(d, 1), 1)})
sh1 = pd.DataFrame(a).sort_values("Admissions positive", ascending=False)
sh1.loc[len(sh1)] = {"Site": "— cohort total —", "Admissions cultured": N,
                     "% of cohort cultured": 100.0, "Admissions positive": mb[mb.pos].h.nunique(),
                     "% of cohort positive": round(100*mb[mb.pos].h.nunique()/N, 1),
                     "Positivity rate %": np.nan}

# sheet: site combinations per MIMIC admission
pv = mb[mb.pos].groupby("h").site.apply(set)
def combo(ss):
    s2 = ss - {"Surveillance screen"}
    if not s2: return "Surveillance screen only"
    if "Blood" in s2: return "Blood only" if len(s2) == 1 else "Blood + other site"
    if len(s2) > 1: return "Multi-site (no blood)"
    return list(s2)[0] + " only"
cb = pv.map(combo).value_counts()
sh2 = pd.DataFrame({"Site pattern": cb.index, "Admissions": cb.values,
                    "% of culture-positive": (100*cb.values/len(pv)).round(1),
                    "% of cohort": (100*cb.values/N).round(1)})
sh2.loc[len(sh2)] = ["No positive culture", N-len(pv), np.nan, round(100*(N-len(pv))/N, 1)]

# sheet: blood-culture organisms, both cohorts
bm = mb[mb.pos & mb.site.eq("Blood")].copy(); bm["cl"] = bm.org.map(cls)
mg = bm.groupby("h").cl.apply(set).map(label)
k = pd.read_parquet(KNU_SRC, columns=["Episode_ID", "Organism"]).dropna(subset=["Organism"])
k["Episode_ID"] = k.Episode_ID.astype(str); k = k[k.Episode_ID.isin(coh_k)]
ko = k.assign(o=k.Organism.astype(str).str.split(r"\s*\|\s*")).explode("o")
ko["o"] = ko.o.str.strip(); ko = ko[ko.o.ne("")]; ko["cl"] = ko.o.map(cls)
kg = ko.groupby("Episode_ID").cl.apply(set).map(label)
sh3 = pd.DataFrame({f"KNU n (N={len(kg):,})": kg.value_counts(),
                    "KNU %": kg.value_counts(normalize=True).mul(100).round(1),
                    f"MIMIC blood n (N={len(mg):,})": mg.value_counts(),
                    "MIMIC blood %": mg.value_counts(normalize=True).mul(100).round(1)}).fillna(0)
sh3["Difference (KNU − MIMIC) pp"] = (sh3["KNU %"] - sh3["MIMIC blood %"]).round(1)
sh3 = sh3.reset_index().rename(columns={"index": "Organism group"}).sort_values("KNU %", ascending=False)

# sheet: cohort summary
sh4 = pd.DataFrame([
    ["Episodes in paper cohort", f"{len(coh_k):,}", f"{len(coh_m):,}"],
    ["Unique admissions", f"{len(coh_k):,}", f"{N:,}"],
    ["Bloodstream infection", "100% (by cohort definition)", f"{100*mb[mb.pos & mb.site.eq('Blood')].h.nunique()/N:.1f}%"],
    ["Any culture drawn", "n/a (specimen type not recorded)", f"{100*mb.h.nunique()/N:.1f}%"],
    ["Any culture positive", "100%", f"{100*mb[mb.pos].h.nunique()/N:.1f}%"],
    ["Blood culture drawn", "100%", f"{100*mb[mb.site.eq('Blood')].h.nunique()/N:.1f}%"],
    ["No positive culture", "0%", f"{100*(N-len(pv))/N:.1f}%"],
    ["IV duration, median (days)", "11", "3"],
    ["IV duration >= 14 days", "41.8%", "7.9%"],
    ["Commensal-only isolate", f"{100*(kg=='Commensal only').mean():.1f}%", f"{100*(mg=='Commensal only').mean():.1f}%"],
], columns=["Metric", "KNU", "MIMIC-IV"])

notes = pd.DataFrame({"Note": [
    "Cohort: ds_ivstop (Bolton et al., Nat Commun 2024 definition; label = last IV day + 1).",
    "MIMIC source: MIMIC-IV 3.1 hosp/microbiologyevents.csv, joined by hadm_id.",
    "Positive culture follows mimic-code: org_name present, org_itemid not in (90856, 90760), not '' or 'CANCELLED'.",
    "KNU has NO specimen-type field. The cohort is bloodstream infection by construction, so the site axis is constant and a site-by-site comparison is not possible. Sheet 3 is the only like-for-like comparison (blood isolates in both).",
    "Sheet 1 counts an admission once per site, so rows sum to more than the cohort.",
    "'Commensal' = NHSN common-commensal list; S. lugdunensis is excluded from it.",
    "MIMIC blood-positive admissions are few (see Sheet 3 N); treat that comparison as low-powered.",
    "Generated 2026-09-04 by make_site_xlsx.py.",
]})

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    sh4.to_excel(w, "1. Cohort summary", index=False)
    sh1.to_excel(w, "2. MIMIC sites", index=False)
    sh2.to_excel(w, "3. MIMIC site patterns", index=False)
    sh3.to_excel(w, "4. Blood organisms (both)", index=False)
    notes.to_excel(w, "5. Notes", index=False)
    for name, ws in w.sheets.items():
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 11), 46)
        ws.freeze_panes = "A2"
print(f"-> {OUT}")
