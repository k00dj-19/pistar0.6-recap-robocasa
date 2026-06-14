#!/bin/bash
#SBATCH --job-name=rc_vf_ind
#SBATCH --qos=own
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# RECAP iteration step (CPU-only): train the multi-task task-conditioned VF on
# D = 4-task demos + accumulated rollouts (failures included), then compute the
# per-task ε_ℓ advantage indicators positionally aligned to the MultiLeRobotDataset
# (demos + rollouts) order for the RECAP policy finetune. VF is a tiny state-only MLP
# (no images/GPU) -> CPU-only job (gres=gpu:0, doesn't touch the GPU quota).
#
# Usage: ITER=1 ROLLOUTS="local/robocasa_rollouts_CloseFridge local/..." \
#          sbatch slurm/vf_indicators_robocasa_mt.sbatch
set -euo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap TOKENIZERS_PARALLELISM=false

ITER=${ITER:-1}
# 4-task demo set (fixed). Rollouts accumulate across iterations (pass via ROLLOUTS env).
DEMOS="pepijn223/robocasa_CloseFridge pepijn223/robocasa_OpenDrawer pepijn223/robocasa_OpenCabinet pepijn223/robocasa_PickPlaceCounterToCabinet"
ROLLOUTS=${ROLLOUTS:?need ROLLOUTS env (space-separated rollout repo_ids accumulated so far)}
OUT=${OUT:-outputs/robocasa/multi_task/vf/iter${ITER}}
PF=${PF:-0.30}
echo "[vf-ind] iter=$ITER out=$OUT pf=$PF"
echo "[vf-ind] demos=$DEMOS"
echo "[vf-ind] rollouts=$ROLLOUTS"

# 1) VF on demos + accumulated rollouts (failures -> VF learns value drops)
python recap/scripts/train_vf_robocasa_multitask.py \
  --root "$PROJ/.lerobot" --device cpu \
  --demos $DEMOS --rollouts $ROLLOUTS \
  --out "$OUT" --epochs 20

# 2) per-task ε_ℓ advantage indicators, positionally aligned to MultiLeRobotDataset order
python recap/scripts/compute_indicators_robocasa_multitask.py \
  --root "$PROJ/.lerobot" --device cpu \
  --demos $DEMOS --rollouts $ROLLOUTS \
  --vf "$OUT/vf.pt" --out "$OUT" --positive_fraction "$PF" --n_step 50

echo "[vf-ind] done -> $OUT (vf.pt + indicators_p$(printf '%02d' $(python -c "print(int($PF*100))")).npz)"
