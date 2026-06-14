#!/bin/bash
#SBATCH --job-name=rc_vlmvf_ind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --qos=own
#SBATCH --time=01:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Stage 2: distributional VF over the VLM features (StateTaskVF, state_dim=H) + per-task
# corrections-style advantage indicators (demo_positive + fail_dense + per-task eps_l).
# Mirrors the PnP corrections-RECAP indicator build EXACTLY, swapping only the critic's
# input from 16-dim proprio -> scene-aware VLM features.
set -euo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap
export TOKENIZERS_PARALLELISM=false

TASK=${TASK:-PickPlaceCounterToCabinet}
OUT=${OUT:-outputs/robocasa/specialist_v2/${TASK}/vlmvf}
# FEATS defaults to OUT/features.npy, but iteration sub-dirs (vlmvf_iter*) reuse the task's
# single shared feature matrix in vlmvf/features.npy -> pass FEATS explicitly.
FEATS=${FEATS:-$OUT/features.npy}
DEMOS="pepijn223/robocasa_${TASK}"
ROLLOUTS=${ROLLOUTS:?need ROLLOUTS env (space-separated rollout repo_ids, npz order)}

test -f "$FEATS" || { echo "[fatal] missing $FEATS (stage-1 extract failed)"; exit 1; }
echo "[sbatch] VLM-VF train + indicators out=$OUT feats=$FEATS"

# (a) distributional VF over VLM features
python recap/scripts/train_vf_robocasa_multitask.py \
  --root .lerobot --demos $DEMOS --rollouts $ROLLOUTS \
  --features "$FEATS" --out "$OUT" \
  --epochs 30 --batch_size 2048 --device cuda

# (b) per-task corrections-style advantage indicators (demo_positive + fail_dense), n_step=50
python recap/scripts/compute_indicators_robocasa_multitask.py \
  --root .lerobot --demos $DEMOS --rollouts $ROLLOUTS \
  --vf "$OUT/vf.pt" --features "$FEATS" --out "$OUT" \
  --positive_fraction 0.30 --n_step 50 --demo_positive --tag_suffix corr_vlmvf \
  --device cuda

echo "[sbatch] done -> $OUT/indicators_p30corr_vlmvf.npz"
