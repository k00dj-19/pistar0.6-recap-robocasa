#!/bin/bash
#SBATCH --job-name=rc_ft4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=720G
#SBATCH --qos=own
#SBATCH --time=02:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# 4-GPU DDP variant of finetune_robocasa.sbatch for a SINGLE task + its rollouts (specialist
# RECAP). Kept RECIPE-IDENTICAL to the 1-GPU baseline so results stay apples-to-apples:
#   effective batch = batch_size(2) x nproc(4) = 8  (== 1-GPU batch 8)
#   steps 6000, lr 2.5e-5, cosine  (unchanged)
# Only the wall-clock changes (~3x faster). Use for the VLM-VF PnP RECAP retrain so it stays
# directly comparable to corrections-RECAP (1-GPU, eff batch 8, 6000 steps).
#   ROLLOUTS=<csv> TASK=PickPlaceCounterToCabinet BASE=<mt_sft> EXPERT=full \
#     sbatch slurm/finetune_robocasa_4gpu.sbatch recap <indicators.npz> <out_dir>
set -euo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

MODE=${1:-recap}
IND=${2:?need indicators npz}
OUT=${3:?need out dir}
# recap MUST be full finetune; EXPERT=full forces it (capacity-matched vs baseline).
EXPERT_FLAG=""
[ "$MODE" = "sft" ] && EXPERT_FLAG="--train_expert_only"
[ "${EXPERT:-}" = "full" ] && EXPERT_FLAG=""
[ "${EXPERT:-}" = "expert" ] && EXPERT_FLAG="--train_expert_only"
TASK=${TASK:-PickPlaceCounterToCabinet}
BASE=${BASE:-outputs/robocasa/multi_task/sft}
echo "[sbatch] 4-GPU mode=$MODE task=$TASK ind=$IND out=$OUT base=$BASE expert=${EXPERT_FLAG:-none} node=$(hostname) cvd=$CUDA_VISIBLE_DEVICES"

if [ -z "${ROLLOUTS:-}" ]; then
  echo "[fatal] 4gpu specialist RECAP expects ROLLOUTS env (combined demos+rollouts path)"; exit 1
fi

torchrun --standalone --nproc_per_node=4 \
  recap/scripts/train_pi05_recap_robocasa.py \
  --root .lerobot --tasks "$TASK" --rollouts "$ROLLOUTS" \
  --base_ckpt "$BASE" \
  --mode "$MODE" \
  --indicators "$IND" \
  --episodes_per_task 0 \
  --steps 6000 \
  --batch_size 2 \
  --num_workers 8 \
  --lr 2.5e-5 \
  --log_every 50 \
  --save_every 1500 \
  $EXPERT_FLAG \
  --out "$OUT"

echo "[sbatch] done mode=$MODE out=$OUT"
