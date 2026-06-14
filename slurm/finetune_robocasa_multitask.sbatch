#!/bin/bash
#SBATCH --job-name=rc_mt_ft
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=720G
#SBATCH --qos=own
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log
# 4-GPU DDP (torchrun) — sub-own cap is 4 GPU / 56 CPU; 4x12=48 CPU fits. Effective
# batch = batch_size x 4, so the same data coverage finishes ~4x faster than 1 GPU.

# MULTI-TASK pi05 finetune over several RoboCasa tasks (MultiLeRobotDataset). This is the
# paper-faithful generalist setting: the VF sees easy + hard tasks so the value/advantage
# distribution is WIDE (single-task CloseFridge was the narrow worst case -> RECAP=SFT null).
#   sbatch slurm/finetune_robocasa_multitask.sbatch sft     # multi-task BC baseline + pi_ref
#   sbatch slurm/finetune_robocasa_multitask.sbatch recap <indicators.npz>   # after rollout+VF
set -euo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

MODE=${1:-sft}
IND=${2:-outputs/robocasa/multi_task/vf/iter1/indicators_p30.npz}
OUT=${3:-outputs/robocasa/multi_task/${MODE}}
# 4-task set. TurnOnMicrowave / CoffeeSetupMug / PickPlaceSinkToCounter dropped: their live
# env samples objects from categories absent in the lightwheel asset pack -> rollout crashes
# ("a cannot be empty"). 3 easy controls (CloseFridge/OpenDrawer/OpenCabinet, pi0.5 ~85-90%)
# + 1 hard PnP treatment (PickPlaceCounterToCabinet, pi0.5 ~40% -> the RECAP headroom task).
TASKS=${TASKS:-CloseFridge,OpenDrawer,OpenCabinet,PickPlaceCounterToCabinet}
# ROLLOUTS: for recap iterations, the accumulated rollout repo_ids (comma-separated) APPENDED
# after the demos — MUST match the order used to build the indicators ($IND). Empty for the
# demos-only SFT baseline.
ROLLOUTS=${ROLLOUTS:-}
# sft baseline uses full-finetune too (capacity-matched vs recap; single-task showed
# full-finetune is what lifts success 40->80%). Drop --train_expert_only.
echo "[sbatch] mode=$MODE tasks=$TASKS rollouts=${ROLLOUTS:-none} out=$OUT node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"

# 4-GPU single-node DDP. steps=8000 x effective_batch 32.
torchrun --standalone --nproc_per_node=4 \
  recap/scripts/train_pi05_recap_robocasa.py \
  --root "$PROJ/.lerobot" \
  --tasks "$TASKS" \
  --rollouts "$ROLLOUTS" \
  --mode "$MODE" \
  --indicators "$IND" \
  --steps 8000 \
  --batch_size 8 \
  --num_workers 8 \
  --lr 2.5e-5 \
  --log_every 100 \
  --save_every 2000 \
  --out "$OUT"

echo "[sbatch] done mode=$MODE out=$OUT"
