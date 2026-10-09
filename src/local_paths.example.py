"""Machine-specific locations of the raw sources, copied to local_paths.py and filled in.

local_paths.py is git-ignored. Anything set here can also be given as an environment
variable of the same name, which takes precedence. Only the stages that read a given
source need its entry: training and all analyses in scripts/ run from the built
tensors alone.
"""

PATHS = {
    # MIMIC-IV v3.1 as distributed by PhysioNet (the directory holding hosp/ and icu/)
    "MIMIC_RAW": "/path/to/MIMIC-IV_3.1",
    # per-admission severity export used to build the MIMIC cohort
    "MIMIC_SRC": "/path/to/MIMIC_severity.parquet",
    # KNU export (unit-corrected: CRP x10, D-Dimer x1000)
    "KNU_SRC": "/path/to/knudata.parquet",
    # preprocessing tree the cohort builders read their intermediate tables from
    "IVOS_V3": "/path/to/preprocessing_tree",
}
