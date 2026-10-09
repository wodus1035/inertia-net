#!/bin/bash
# Train the 15 cross-validation models of the manuscript (5 folds x seeds 42/43/44),
# four at a time across GPUs 0-3. Run from anywhere; paths resolve to the repository.
REPO=$(cd "$(dirname "$0")/.." && pwd)
SC=${IVOS_SCRATCH:-/tmp/inertia_scratch}
PY=${PYTHON:-python}
mkdir -p "$SC"
cd "$REPO"
i=0
for f in 0 1 2 3 4; do for sd in 42 43 44; do
  env -u KNU_SRC -u MIMIC_SRC \
      IVOS_DATAS=${IVOS_DATAS:-$REPO/data/ds_nocap} \
      IVOS_SPLIT_FILE=${IVOS_SPLIT_FILE:-$REPO/data/ds_nocap/split_patients.parquet} \
      IVOS_OUTPUTS=$SC/out_cv_f${f}_s${sd} PYTHONUNBUFFERED=1 \
      setsid $PY -u -m src.train_inertia_net \
      --gpu $((i%4)) --tag cv_f${f}_s${sd} --seed $sd --fold $f --n_folds 5 --cv_seed 42 \
      --d_model 64 --num_layers 2 --nhead 4 --dropout 0.2 \
      --lr 5e-4 --lambda_mmd 0.8 --smoothing 0.0 \
      --epochs 100 --patience 15 --wandb --tb \
      > $SC/train_cv_f${f}_s${sd}.log 2>&1 < /dev/null &
  i=$((i+1)); sleep 3
done; done
echo "launched $i"
