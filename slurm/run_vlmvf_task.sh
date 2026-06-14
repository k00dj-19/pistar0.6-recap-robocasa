#!/bin/bash
# Per-task scene-aware (VLM) value-function RECAP chain. Reads the rollout repo order from
# that task's corrections indicators npz (authoritative, for positional alignment), then:
#   extract (4-GPU) -> VLM-VF + indicators (1-GPU) -> RECAP retrain (4-GPU) -> eval n=50.
# Usage: bash slurm/run_vlmvf_task.sh <TASK>
set -euo pipefail
cd /home/nas_main/dongjinkim/pi_06_star
TASK=${1:?need task name, e.g. CloseFridge}
PY=.venv_robocasa/bin/python

NPZ=outputs/robocasa/specialist_v2/${TASK}/vf_iter3/indicators_p30corr.npz
test -f "$NPZ" || { echo "[fatal] missing corrections npz for $TASK: $NPZ"; exit 1; }
ROLLOUTS_SPACE=$($PY -c "import numpy as np; print(' '.join(np.load('$NPZ',allow_pickle=True)['repo_ids'].tolist()[1:]))")
ROLLOUTS_CSV=$(echo "$ROLLOUTS_SPACE" | tr ' ' ',')

VLMVF=outputs/robocasa/specialist_v2/${TASK}/vlmvf
RECAP_OUT=outputs/robocasa/specialist_v2/${TASK}/recap_vlmvf

J1=$(ROLLOUTS="$ROLLOUTS_SPACE" TASK="$TASK" \
     sbatch --parsable slurm/extract_vlmvf_robocasa.sbatch)
J2=$(ROLLOUTS="$ROLLOUTS_SPACE" TASK="$TASK" \
     sbatch --parsable --dependency=afterok:$J1 slurm/vlmvf_train_indicators.sbatch)
J3=$(ROLLOUTS="$ROLLOUTS_CSV" TASK="$TASK" BASE=outputs/robocasa/multi_task/sft EXPERT=full \
     sbatch --parsable --dependency=afterok:$J2 slurm/finetune_robocasa_4gpu.sbatch \
     recap "$VLMVF/indicators_p30corr_vlmvf.npz" "$RECAP_OUT")
J4=$(SEED=5000 sbatch --parsable --dependency=afterok:$J3 slurm/eval_robocasa.sbatch \
     "$RECAP_OUT" positive "$TASK" 50 10)
echo "$TASK : $J1(extract) -> $J2(vf+ind) -> $J3(recap4) -> $J4(eval)"
