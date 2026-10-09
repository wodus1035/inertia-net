# Inertia-Net

Inertia-Net learns one **readiness score** for intravenous-to-oral (IV-to-oral)
antibiotic switching that is shared across institutions, and confines every
institution-specific difference in *when a switch is actually made* to a few scalar
parameters. Because the score axis is shared, switching behavior at two hospitals can
be compared at matched patient state instead of through raw switching rates, which
mostly reflect how long intravenous courses last locally.

## Study in brief

Two cohorts of intravenous antibiotic courses with an observed IV-to-oral switch:

| Cohort | Setting | Episodes | Patients | Median IV duration |
| --- | --- | ---: | ---: | ---: |
| KNU | Sepsis with bacteremia, two Korean hospitals (KNUH, KNUCH), 2013–2024 | 9,503 | 8,390 | 10 days |
| MIMIC-IV v3.1 | ICU stays, Beth Israel Deaconess Medical Center, 2008–2022 | 8,712 | 7,560 | 3 days |

Each day of an intravenous course before the switch is an *anchor-day*. The model
reads the preceding 72 h of vital signs and laboratory values and predicts whether the
switch occurs within 1, 2 or 3 days.

Main results on the held-out test split (15 models, five folds × three seeds):

- **Discrimination is maintained in both cohorts** with one shared representation:
  AUROC 0.75–0.84 at KNU and 0.78–0.83 in MIMIC-IV. A model trained on KNU alone loses
  0.13–0.16 AUROC when applied to MIMIC-IV without adaptation.
- **At the same readiness score, switching is more frequent in MIMIC-IV.** At the
  median score the cohort odds ratio is 3.00, 3.19 and 3.49 at 1, 2 and 3 days. The
  divergence persists across ICU-stratified, Sepsis-3 and blood-culture-matched
  sensitivity analyses.
- **Switching-threshold gap.** To reach a 50 % probability of switching, KNU
  anchor-days need a score 0.277, 0.465 and 0.603 units higher than MIMIC-IV
  anchor-days at 1, 2 and 3 days (0.46, 0.77 and 1.00 SD of the pooled score).
- **At high modeled readiness** (top quintile), 59.5 % of KNU anchor-days versus
  24.5 % in MIMIC-IV are not followed by a switch within 3 days, and when KNU does
  switch, 86.1 % of those episodes are discharged within one day of the switch.

## Model

Irregular clinical time series are encoded as **STraTS-style triplets**
(Tipirneni & Reddy, *TKDD* 2022): each observation becomes a `(time, variable, value)`
token, so nothing is imputed or interpolated. Vital signs are binned at 2 h and
laboratory values at 24 h over a 72 h window.

A single Transformer encoder (2 layers, `d_model` 64, 4 heads, attention pooling,
FiLM code-value fusion) and a single linear head produce the scalar readiness score
`s(z)`. Each cohort `d` and horizon `h` carries only a cutpoint `c[d,h]`:

```
logit[d, h] = tau * ( s(z) - c[d, h] ),        tau = 3.0
```

Training is joint over the two cohorts with class-weighted binary cross-entropy and a
multi-kernel MMD term (weight 0.8) that aligns the score distributions. The cutpoints
absorb each cohort's switching prevalence during training so that the shared score is
not forced to encode it. Every reported comparison is then made on `s(z)` itself:

- **Score-conditional divergence** — observed switching by decile of the pooled score,
  and a logistic model with a cohort-by-score interaction (patient cluster-robust SE).
- **Switching-threshold gap** — a two-parameter logistic map of `s(z)` to switching is
  fitted per cohort on the held-out test split, and the gap is the difference in the
  score at which each cohort reaches 50 % (`c50 = -b/a`), with patient-clustered
  bootstrap intervals.
- **Calibration** — Platt parameters fitted on each fold's validation split and applied
  to the test split (Brier, ECE, slope, intercept).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Model (`model.py`), training (`train_inertia_net.py`), data loading and configuration |
| `data_build/` | Cohort construction and tensor building for both cohorts |
| `scripts/` | Analysis and evaluation scripts; `common.py` holds shared paths, checkpoint loading, 15-model scoring and statistics |
| `results/` | Numerical results behind the manuscript tables (aggregate values only, no patient-level data) |
| `models/` | The 15 cross-validation checkpoints reported in the manuscript (`cv_f{0..4}_s{42,43,44}/`) |

## Reproducing the results

The 15 checkpoints are in `models/cv_f{fold}_s{seed}/`. The scripts also need the
built tensors, which are not distributed. Scores on the held-out test split are cached once in
`results/scores15_test.npz` by `scripts/common.py`.

```bash
python scripts/cv_eval.py                        # score the 15 models on the test split
python scripts/cv_report.py                      # discrimination, gap and calibration
```

The other scripts in `scripts/` are named after what they produce, and each one
writes or prints the values behind one manuscript table. Some of
them read a per-episode table of admission, switch and discharge dates
(`episode_table_{knu,mimic}.parquet`) that is derived from the source records and,
like the tensors, is not distributed.

`results/` also holds analyses run for review but not reported in the manuscript:
`cutpoint_ablation.csv`, `organism_adjusted.txt`, and `pooled_baseline.json` /
`pooled_indicator.json`.

## Requirements

Python 3.10 with:

```
torch >= 2.7      numpy >= 1.24     pandas >= 2.0
scikit-learn      statsmodels       pyarrow
matplotlib        xgboost           # XGBoost baseline only
```

A CUDA GPU is recommended; training one fold takes roughly 40 min on an RTX A6000.
Training uses `cudnn.benchmark`, so runs with the same seed are not bit-reproducible.
The manuscript therefore reports every quantity across the 15 models rather than from
a single run.

## Data

**Neither cohort is redistributable and neither is included in this repository.**

| Cohort | Source | Access |
| --- | --- | --- |
| MIMIC-IV v3.1 | Beth Israel Deaconess Medical Center | PhysioNet, credentialed users |
| KNU | Kyungpook National University Hospital and Chilgok Hospital | Institutional approval required |

The 15 cross-validation checkpoints behind every number in the manuscript are
included in `models/`. Each file holds the encoder and score-head weights, the six
cutpoints, the Platt parameters and the per-cohort standardization statistics, and
carries no patient-level data. The 24-hour and 48-hour sensitivity models are not
included.

### Cohort definition

An episode is a hospital stay receiving intravenous antibiotics followed by an observed
oral switch. The MIMIC-IV cohort follows Bolton et al. (*Nat Commun* 2024): for each
stay the intravenous and oral administrations are each collapsed to one contiguous span,
both routes must be present, and `iv_stop <= po_stop`. The switch day is the day after
the last intravenous administration (`iv_stop + 1`). Unlike the original definition we
impose **no upper limit on intravenous duration**, and the same rule is applied to KNU,
whose cohort consists of sepsis patients with a positive blood culture. Anchor-days are
every recorded day before the switch day; labels are whether the switch occurs within
1, 2 or 3 days.

### Building the tensors

`data_build/` reconstructs the cohorts from the source tables and writes the arrays the
model consumes. Point it at your own copies of the sources by copying
`src/local_paths.example.py` to `src/local_paths.py` and filling in the entries, or by
setting the same names as environment variables. That file is git-ignored, and only the
cohort-building and raw-data steps read it: training and everything in `scripts/` run
from the built tensors. Then:

```bash
python data_build/build_mimic_nocap.py     # MIMIC-IV
python data_build/build_knu_matched.py     # KNU, same rules
python data_build/make_split_nocap.py      # patient-level 70/15/15 split
```

Each cohort produces `<name>_tensors.npz` (triplets, static features, labels) and
`<name>_meta.parquet` (one row per anchor-day). The split is by patient, so no patient
appears in more than one partition.

## Training and evaluation

```bash
export IVOS_DATAS=$PWD/data/ds_nocap
export IVOS_SPLIT_FILE=$PWD/data/ds_nocap/split_patients.parquet
export IVOS_SCRATCH=$PWD/scratch          # working directory for launch scripts

# One cross-validation model (fold 0, seed 42); scripts/launch_cv.sh runs all 15
python -m src.train_inertia_net --gpu 0 --tag cv_f0_s42 --seed 42 \
  --fold 0 --n_folds 5 --cv_seed 42 \
  --d_model 64 --num_layers 2 --nhead 4 --dropout 0.2 \
  --lr 5e-4 --lambda_mmd 0.8 --epochs 100 --patience 15
# add --shared_cutpoints for the shared-cutpoint ablation

# Held-out test evaluation of the 15 models: discrimination, gap, calibration
python scripts/cv_eval.py
python scripts/cv_report.py

# Place an additional institution on the shared axis without retraining
python scripts/place_institution.py --ckpt models/cv_f0_s42/inertia_net_model.pth --data <tensors.npz>
```

Hyperparameters were selected by a 24-configuration search over `d_model`, depth,
learning rate, dropout and MMD weight (`results/hpo_trials.json`, `scripts/hpo_gap.py`).

## Reference implementations consulted

- [WilliamBolton/iv_to_oral](https://github.com/WilliamBolton/iv_to_oral) — MIMIC-IV cohort and label definition
- [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) — Sepsis-3 and antibiotic concepts
- [sindhura97/STraTS](https://github.com/sindhura97/STraTS) — triplet encoding of irregular time series

## Ethics

The KNU study was approved by the Institutional Review Boards of Kyungpook National
University Hospital (KNUH 2022-05-200-005) and Kyungpook National University Chilgok
Hospital (KNUCH 2024-07-105-005), which waived informed consent because of the
retrospective design. MIMIC-IV was used under the PhysioNet data use agreement.
