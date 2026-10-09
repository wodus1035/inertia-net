"""Physiologic clip ranges for the raw channels, keyed by canonical quantity with
per-cohort column-name aliases, so a cohort that spells a channel differently reuses
the same range. Applied in data.load_domain."""

import numpy as np

# canonical_name -> {"range": (lo, hi), "aliases": [every observed column spelling]}
VITAL_REGISTRY = {
    "systolic_bp":   {"range": (30, 300), "aliases": ["SBP"]},
    "diastolic_bp":  {"range": (20, 200), "aliases": ["DBP"]},
    "heart_rate":    {"range": (20, 300), "aliases": ["Pulse", "HR"]},
    "resp_rate":     {"range": (0, 80),   "aliases": ["Resp_Rate", "Resp", "RR"]},
    "body_temp":     {"range": (30, 45),  "aliases": ["Body_Temp", "BodyTemp", "Temp"]},
    "spo2":          {"range": (30, 100), "aliases": ["SpO2"]},
    "resp_support":  {"range": (0, 1),    "aliases": ["resp_support", "O2_Apply"]},
    "fio2":          {"range": (0, 100),  "aliases": ["FiO2"]},         
    "o2_amount":     {"range": (0, 100),  "aliases": ["O2_Amount"]},
    "flow_rate":     {"range": (0, 100),  "aliases": ["Flow_Rate", "Flow"]},
    "avpu":          {"range": (0, 3),    "aliases": ["AVPU"]},         
}

LAB_REGISTRY = {
    "wbc":         {"range": (0, 100),   "aliases": ["WBC"]},
    "platelets":   {"range": (0, 1500),  "aliases": ["Platelets", "Plt"]},
    "hemoglobin":  {"range": (0, 25),    "aliases": ["Hemoglobin", "Hgb"]},
    "lymphocytes": {"range": (0, 100),   "aliases": ["Lymphocytes", "Lymph"]},
    "neutrophils": {"range": (0, 100),   "aliases": ["Neutrophils", "Neut"]},
    "creatinine":  {"range": (0, 20),    "aliases": ["Creatinine", "Cr"]},
    "bun":         {"range": (0, 200),   "aliases": ["BUN"]},
    "sodium":      {"range": (80, 200),  "aliases": ["Sodium", "Na"]},
    "potassium":   {"range": (0, 10),    "aliases": ["Potassium", "K"]},
    "crp":         {"range": (0, 500),   "aliases": ["CRP"]},
    "ph":          {"range": (6.5, 8.0), "aliases": ["pH"]},
    "pco2":        {"range": (0, 200),   "aliases": ["pCO2"]},
    "po2":         {"range": (0, 700),   "aliases": ["pO2"]},
    "hco3":        {"range": (0, 60),    "aliases": ["HCO3"]},
    "total_co2":   {"range": (0, 60),    "aliases": ["Total_CO2", "TCO2"]},
    "o2sat":       {"range": (0, 100),   "aliases": ["O2Sat", "O2_Sat"]},
}


def _build_alias_lookup(registry):
    """Flatten to {alias: (lo, hi)}."""
    lookup = {}
    for canon, spec in registry.items():
        for alias in spec["aliases"]:
            if alias in lookup:
                raise ValueError(
                    f"alias '{alias}' is registered under multiple canonical "
                    f"names — check the registry for a copy-paste duplicate."
                )
            lookup[alias] = spec["range"]
    return lookup


def clip_domain_by_registry(X, cols, registry, verbose=True):
    """X: (N, T, C); cols: that cohort's (C,) column names. Channels with no registry
    entry are left unclipped."""
    alias_lookup = _build_alias_lookup(registry)
    out = X.copy()
    unmatched = []
    for i, name in enumerate(cols):
        if name in alias_lookup:
            lo, hi = alias_lookup[name]
            out[..., i] = np.clip(out[..., i], lo, hi)
        else:
            unmatched.append(str(name))
    if verbose and unmatched:
        print(f"[clip_domain_by_registry] no registry entry for channels "
              f"(left unclipped, consider adding an alias): {unmatched}")
    return out
