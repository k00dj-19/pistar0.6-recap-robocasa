#!/bin/bash
#SBATCH --job-name=rc_collect_lerobot
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# RECAP factor 5: collect auto-labeled RoboCasa rollouts from a (RECAP-finetuned)
# pi05 policy into a full LeRobotDataset (images + state + action + task) plus a
# side episode_success.npz. Same NGC container / GLVND / offline / dataset_root
# setup as collect_rollouts_robocasa.sbatch — only the script and OUT layout differ.
# N_EP / SEED are parametrized and OUT is per-seed so several jobs fan out across
# GPUs with distinct seeds + non-colliding output dirs.
#
# GPU: 1 (pi05 fits on one B200) + --qos=own. Before submitting confirm
# own-QOS GPU usage stays within your quota:
#   squeue -u $USER -o "%.10i %.12j %.8T %b %.10q %R"
#
# Usage: N_EP=20 SEED=1000 sbatch slurm/collect_rollouts_lerobot_robocasa.sbatch <ckpt_dir> <task> <condition>

set -euo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

# HF_HOME set UNCONDITIONALLY (cluster profile pre-exports the shared path; a `:-`
# fallback would not override it -> shared-cache paligemma .locks PermissionError).
export HF_HOME="$PROJ/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
# GLVND EGL frontend (vendored jammy libglvnd) + CUDA NPP for torchcodec.
GLVND_LIB="$PROJ/glvnd/root/usr/lib/x86_64-linux-gnu"
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$GLVND_LIB:$SP/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

CKPT=${1:-outputs/robocasa/single_task/sft_full}
TASK=${2:-CloseFridge}
COND=${3:-none}
N_EP=${N_EP:-20}
SEED=${SEED:-1000}
# CloseFridge snapshot (shared cache) -> dataset stats for the pi05 processors,
# passed as --dataset_root so no HF lookup happens under offline mode.
CF=${CF:-/home/nas_main/.cache/huggingface/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4}
TAG=$(basename "$CKPT")_${TASK}_${COND}_s${SEED}
# Per-seed OUT dir so concurrent jobs don't collide. Must NOT pre-exist
# (LeRobotDataset.create requires a fresh dir).
OUT=${OUT:-$PROJ/outputs/rollouts_lerobot/${TAG}}
# REPO_ID must be unique per dataset dir (MultiLeRobotDataset resolves datasets by repo_id).
# Default derives from task; override for extra seeds / iterations so they don't collide.
REPO_ID=${REPO_ID:-local/robocasa_rollouts_${TASK}}

echo "[collect] ckpt=$CKPT task=$TASK cond=$COND n_ep=$N_EP seed=$SEED repo_id=$REPO_ID node=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset} out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader || true

srun python recap/scripts/collect_rollouts_lerobot_robocasa.py \
  --ckpt "$CKPT" \
  --task "$TASK" \
  --condition "$COND" \
  --n_episodes "$N_EP" \
  --n_envs 1 \
  --seed "$SEED" \
  --repo_id "$REPO_ID" \
  --dataset_root "$CF" \
  --out "$OUT"

echo "[collect] done $TAG -> $OUT"
