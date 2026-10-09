"""Patient-level held-out split.

All episodes of a patient go to the same split, so nothing leaks between train
and test. Written to SPLIT_FILE with columns: domain, patient, split.

    python -m src.split      # build and save the split
"""
import numpy as np
import pandas as pd

from .config import (MIMIC_TENS, KNU_TENS, SPLIT_FILE, SPLIT_SEED, TRAIN_FRAC, VAL_FRAC,
                     patient_of)


def _assign(patients, seed):
    """Shuffle patients and label them train/val/test."""
    uniq = np.array(sorted(set(patients)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq); n_tr = int(n * TRAIN_FRAC); n_va = int(n * VAL_FRAC)
    lab = np.array(['test'] * n, dtype=object)
    lab[:n_tr] = 'train'; lab[n_tr:n_tr + n_va] = 'val'
    return pd.DataFrame({'patient': uniq, 'split': lab})


def _domain_patients(tens_path):
    import numpy as _np
    z = _np.load(str(tens_path), allow_pickle=True)
    eids = pd.Series(z['episode_id'].astype(str))
    return patient_of(eids).unique()


def build_split():
    rows = []
    for dom, path in [('mimic', MIMIC_TENS), ('knu', KNU_TENS)]:
        pats = _domain_patients(path)
        df = _assign(pats, SPLIT_SEED)
        df.insert(0, 'domain', dom)
        vc = df['split'].value_counts()
        print(f"[{dom}] patients={len(df):,}  "
              f"train={vc.get('train',0):,} val={vc.get('val',0):,} test={vc.get('test',0):,}")
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SPLIT_FILE, index=False)
    print(f"saved -> {SPLIT_FILE}")
    return out


def load_split():
    return pd.read_parquet(SPLIT_FILE)


def sample_split(domain, episode_ids):
    """Map an array of episode_ids to their patients' split labels."""
    sp = load_split()
    m = dict(zip(sp[sp['domain'] == domain]['patient'], sp[sp['domain'] == domain]['split']))
    pats = patient_of(pd.Series(episode_ids).astype(str)).to_numpy()
    return np.array([m.get(p, 'test') for p in pats], dtype=object)


if __name__ == "__main__":
    build_split()
