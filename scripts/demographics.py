"""Supplementary baseline-characteristics and outcomes table, KNU vs MIMIC-IV.
Episode-level values from the same tensor cohort used for training; outcomes come from the
episode_table built by reescalation_and_outcomes. Writes LaTeX rows to stdout."""
import numpy as np, pandas as pd
from common import CACHE, DATA, REPO, rule
from src.config import KNU_SRC, MIMIC_SRC, patient_of
from src.data import load_domain, masks_of
R=f"{REPO}/results/REESCALATION_NOCAP"
SEP=set(pd.read_csv(f"{REPO}/data/sepsis3_repro.csv").query("sepsis3_repro").Episode_ID.astype(str))
BL=set(pd.read_parquet(f"{CACHE}/mimic_blood_positive.parquet").Episode_ID.astype(str))
def ymd(v): return pd.to_datetime(v.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
rows={}
for dom in ("knu","mimic"):
    d=load_domain(dom); eps=np.asarray(d["eps"]).astype(str); n_ep=len(set(eps))
    ep=pd.DataFrame({"ep":eps,"age":d["static"][:,0],"male":d["static"][:,1],"delta":d["delta"]}).groupby("ep").agg(age=("age","first"),male=("male","first"),n=("delta","size"),course=("delta","max"))
    et=pd.read_parquet(f"{R}/episode_table_{dom}.parquet"); et["Episode_ID"]=et.Episode_ID.astype(str); et=et.set_index("Episode_ID").reindex(ep.index)
    src=KNU_SRC if dom=="knu" else MIMIC_SRC
    cols=["Episode_ID","Treatment_Result"]+(["Admission_Date","Discharge_Date"] if dom=="knu" else ["hospital_expire_flag"])
    raw=pd.read_parquet(src,columns=cols); raw["Episode_ID"]=raw.Episode_ID.astype(str); raw=raw[raw.Episode_ID.isin(ep.index)].drop_duplicates("Episode_ID").set_index("Episode_ID").reindex(ep.index)
    tr=raw.Treatment_Result.astype(str).str.lower()
    death=int(et.died.fillna(0).sum()); readm=et.readmit_30d.mean(); los=et.los_total; lps=et.los_post_switch
    sw=(pd.to_datetime(et.switch_terminal)-pd.to_datetime(et.discharge)).dt.days if dom=="knu" else None
    r=dict(n_ep=n_ep,n_pat=patient_of(pd.Series(eps)).nunique(),age=(ep.age.mean(),ep.age.std()),male=(int(ep.male.sum()),ep.male.mean()),
           anchors=(len(eps),ep.n.median(),ep.n.quantile(.25),ep.n.quantile(.75)),
           los=(los.median(),los.quantile(.25),los.quantile(.75)),lps=(lps.median(),lps.quantile(.25),lps.quantile(.75)),
           death=(death,death/n_ep),readm=readm,transfer=tr.str.contains("transfer").mean(),dama=tr.str.contains("dama").mean(),
           sep=(len(set(ep.index)&SEP),len(set(ep.index)&SEP)/n_ep) if dom=="mimic" else (n_ep,1.0),
           blood=(len(set(ep.index)&BL),len(set(ep.index)&BL)/n_ep) if dom=="mimic" else (n_ep,1.0),
           first_tr=(int(et.switch_first.notna().sum()),et.switch_first.notna().mean()))
    rows[dom]=r
K,M=rows["knu"],rows["mimic"]
rule("Supplementary Table — Baseline characteristics and outcomes (LaTeX)")
L=lambda a,b: f"{a} & {b} \\\\"
print(L("Episodes, $n$",f"{K['n_ep']:,} & {M['n_ep']:,}")); print(L("Unique patients, $n$",f"{K['n_pat']:,} & {M['n_pat']:,}"))
print(L("Anchor-days, $n$ (per episode, median [IQR])",f"{K['anchors'][0]:,} ({K['anchors'][1]:.0f} [{K['anchors'][2]:.0f}--{K['anchors'][3]:.0f}]) & {M['anchors'][0]:,} ({M['anchors'][1]:.0f} [{M['anchors'][2]:.0f}--{M['anchors'][3]:.0f}])"))
print(L("Age, years, mean $\\pm$ SD",f"{K['age'][0]:.1f} $\\pm$ {K['age'][1]:.1f} & {M['age'][0]:.1f} $\\pm$ {M['age'][1]:.1f}"))
print(L("Male sex, $n$ (\\%)",f"{K['male'][0]:,} ({100*K['male'][1]:.1f}) & {M['male'][0]:,} ({100*M['male'][1]:.1f})"))
print(L("Blood-culture-confirmed infection, $n$ (\\%)",f"{K['blood'][0]:,} (100.0) & {M['blood'][0]:,} ({100*M['blood'][1]:.1f})"))
print(L("Met Sepsis-3 criteria, $n$ (\\%)",f"{K['sep'][0]:,} (100.0)$^{{a}}$ & {M['sep'][0]:,} ({100*M['sep'][1]:.1f})$^{{b}}$"))
print(L("Hospital length of stay, days, median [IQR]",f"{K['los'][0]:.0f} [{K['los'][1]:.0f}--{K['los'][2]:.0f}] & {M['los'][0]:.0f} [{M['los'][1]:.0f}--{M['los'][2]:.0f}]"))
print(L("Inpatient days after the switch, median [IQR]",f"{K['lps'][0]:.0f} [{K['lps'][1]:.0f}--{K['lps'][2]:.0f}] & {M['lps'][0]:.0f} [{M['lps'][1]:.0f}--{M['lps'][2]:.0f}]"))
print(L("Identifiable first IV-to-oral transition, $n$ (\\%)",f"{K['first_tr'][0]:,} ({100*K['first_tr'][1]:.1f}) & {M['first_tr'][0]:,} ({100*M['first_tr'][1]:.1f})"))
print(L("In-hospital death, $n$ (\\%)",f"{K['death'][0]:,} ({100*K['death'][1]:.1f}) & {M['death'][0]:,} ({100*M['death'][1]:.1f})"))
print(L("Readmission within 30 days, \\%",f"{100*K['readm']:.1f} & {100*M['readm']:.1f}"))
print(L("Discharged by transfer, \\%",f"{100*K['transfer']:.1f} & {100*M['transfer']:.1f}"))
print(L("Discharge against medical advice, \\%",f"{100*K['dama']:.1f} & {100*M['dama']:.1f}"))
