#!/bin/bash
SC=${IVOS_SCRATCH:-/tmp/inertia_scratch}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
Q=$SC/hpo_nc_queue.txt
${PYTHON:-python} - <<'PY' > $Q
import json
SC=${IVOS_SCRATCH:-/tmp/inertia_scratch}
for i,t in enumerate(json.load(open(f"{SC}/hpo_trials.json"))):
    print(f"{i} {t['d_model']} {t['num_layers']} {t['lr']} {t['dropout']} {t['lambda_mmd']} {t['weight_decay']}")
PY
touch $Q.lock
worker () {
  local gpu=$1
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i '1d' $Q")
    [ -z "$line" ] && break
    set -- $line; T=$1; DM=$2; NL=$3; LR=$4; DO=$5; MMD=$6; WD=$7
    tag=hn_t${T}
    [ -f $SC/out_${tag}/inertia_net/${tag}_fold0/inertia_net_model.pth ] && continue
    env -u KNU_SRC -u MIMIC_SRC \
      IVOS_DATAS=$SC/ds_nocap IVOS_SPLIT_FILE=$SC/ds_nocap/split_patients.parquet \
      IVOS_OUTPUTS=$SC/out_${tag} PYTHONUNBUFFERED=1 \
      ${PYTHON:-python} -u -m src.train_inertia_net \
      --gpu $gpu --tag $tag --seed 42 --fold 0 --n_folds 5 --cv_seed 42 \
      --d_model $DM --num_layers $NL --nhead 4 --dropout $DO \
      --lr $LR --lambda_mmd $MMD --weight_decay $WD \
      --smoothing 0.0 \
      --epochs 100 --patience 10 --wandb --tb \
      > $SC/in_${tag}.log 2>&1 < /dev/null
  done
}
for g in 0 1 2 3; do for s in 1 2 3; do worker $g & done; done
wait
echo "HPO_NC_DONE $(date)"
