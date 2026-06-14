#!/bin/bash
#SBATCH --job-name=rc_vlmfeat
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=720G
#SBATCH --qos=own
#SBATCH --time=04:30:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Stage 1 of the scene-aware (VLM) value function experiment for PnP: run the frozen
# mt_sft PaliGemma backbone over every demo+rollout frame and dump masked-mean-pooled
# prefix features -> features.npy. 4-GPU DDP (forward-only) so a 423k-frame pass finishes
# fast. Output feeds train_vf (--features) + compute_indicators (--features).
set -euo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

# TASK-generic. ROLLOUTS (space-separated repo_ids) MUST match the order in that task's
# indicators_p30corr.npz repo_ids[1:] — positional alignment is mandatory. The per-task
# launcher (run_vlmvf_task.sh) reads it from the npz and exports it.
TASK=${TASK:-PickPlaceCounterToCabinet}
BASE=${BASE:-outputs/robocasa/multi_task/sft}
OUT=${OUT:-outputs/robocasa/specialist_v2/${TASK}/vlmvf}
DEMOS="pepijn223/robocasa_${TASK}"
ROLLOUTS=${ROLLOUTS:?need ROLLOUTS env (space-separated rollout repo_ids, npz order)}

echo "[sbatch] VLM feature extract task=$TASK base=$BASE out=$OUT node=$(hostname) cvd=$CUDA_VISIBLE_DEVICES"
torchrun --standalone --nproc_per_node=4 \
  recap/scripts/extract_vlm_features_robocasa.py \
  --root .lerobot --base_ckpt "$BASE" \
  --demos $DEMOS --rollouts $ROLLOUTS \
  --batch_size 48 --num_workers 10 \
  --out "$OUT"
echo "[sbatch] done -> $OUT/features.npy"
