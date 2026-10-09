"""Paths, feature lists, tensor window spec and the patient-level held-out split.

Every stage of the pipeline reads its configuration from here. Feature lists and
the window can be overridden with the IVOS_* environment variables.
"""
import os
from pathlib import Path

try:                                  # git-ignored; see local_paths.example.py
    from .local_paths import PATHS as _LOCAL
except ImportError:
    _LOCAL = {}


def _src(name):
    """Raw-source location: environment variable, then local_paths.py. The placeholder
    makes an unset source fail with its own name rather than a stray path."""
    return os.environ.get(name) or _LOCAL.get(name) or f"<set {name}>"

IVOS_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = Path(os.environ.get("IVOS_OUTPUTS", IVOS_ROOT / "output_model"))
DATAS   = Path(os.environ.get("IVOS_DATAS",   IVOS_ROOT / "datasets"))

# Raw sources, read only by data_build/ when rebuilding the tensors.
MIMIC_SRC = _src("MIMIC_SRC")
KNU_SRC   = _src("KNU_SRC")          # unit-corrected export (CRP x10, D-Dimer x1000)

# Features. Only column names present in the raw source parquet are valid.
def _env_list(name, default):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else default

VITALS = _env_list("IVOS_VITALS", [
    'SBP', 'DBP', 'Pulse', 'Resp_Rate', 'Body_Temp', 'SpO2',
    'resp_support', 'FiO2', 'O2_Amount', 'Flow_Rate', 'AVPU'])
# resp_support and AVPU are derived in preprocessing, keyed by these names; the
# rest are raw values. A derived channel is only built if its name is listed.
LABS   = _env_list("IVOS_LABS", [
    'WBC', 'Platelets', 'Hemoglobin', 'Lymphocytes', 'Neutrophils',
    'Creatinine', 'BUN', 'Sodium', 'Potassium', 'CRP',
    'pH', 'pCO2', 'pO2', 'HCO3', 'Total_CO2', 'O2_Sat'])
LAB_COLS    = LABS
STATIC_NUMERIC_IDX = [0]                  # z-score age only; sex_M is binary

# Tensor window. Invariant: N_BINS * BIN_HOURS == INPUT_DAYS * 24.
INPUT_DAYS = int(os.environ.get("IVOS_INPUT_DAYS", 2))    # days up to and including the anchor day
BIN_HOURS  = int(os.environ.get("IVOS_BIN_HOURS", 2))
N_BINS     = int(os.environ.get("IVOS_N_BINS", INPUT_DAYS * 24 // BIN_HOURS))
WINDOW_TAG = f"d{INPUT_DAYS}_b{N_BINS}x{BIN_HOURS}h"

# Cohort filter: drop episodes whose record span exceeds this (bad admission links).
MAX_EPISODE_SPAN_DAYS = 60

# Held-out split is assigned per Patient_ID, not per episode, to avoid leakage
# (Episode_ID = "PatientID|date"). Independent of window/features, so one file is
# shared across all sweeps. Path(), not str: split.py mkdirs its parent.
SPLIT_FILE  = Path(os.environ.get("IVOS_SPLIT_FILE", DATAS / "split_patients.parquet"))
SPLIT_SEED  = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15

# Outputs are tagged with WINDOW_TAG so sweeps do not collide; SPLIT_FILE is shared.
MIMIC_TENS   = DATAS / f"mimic_tensors_{WINDOW_TAG}.npz"
MIMIC_META   = DATAS / f"mimic_meta_{WINDOW_TAG}.parquet"
KNU_TENS     = DATAS / f"knu_tensors_{WINDOW_TAG}.npz"
KNU_META     = DATAS / f"knu_meta_{WINDOW_TAG}.parquet"


def patient_of(episode_id):
    """Episode_ID ('PatientID|date') -> PatientID. Accepts array, Series or scalar."""
    import pandas as pd
    s = pd.Series(episode_id, dtype="string") if not isinstance(episode_id, pd.Series) else episode_id
    return s.astype(str).str.split("|").str[0]
