#!/bin/bash
#SBATCH --job-name=rc_ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --qos=own
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Our OWN π0.5 finetune on RoboCasa CloseFridge — the honest baseline that retires the
# 0% public-checkpoint dead-end (quant ckpt too weak; ruiname doesn't know CloseFridge).
# One independent experiment per mode.
#   sbatch slurm/finetune_robocasa.sbatch sft     # plain BC baseline (expert-only, fast)
#   sbatch slurm/finetune_robocasa.sbatch recap   # advantage-conditioned (full finetune)
#
# Scope note (per critic review): on single-task, all-success CloseFridge demos the
# advantage encodes only relative speed, so RECAP is expected to ~tie SFT. The deliverable
# here is (a) a working non-zero baseline and (b) a confirmatory generalization datapoint,
# NOT a RoboCasa improvement number.
set -euo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export HF_HOME="$PROJ/.hf"
export HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

MODE=${1:-sft}
IND=${2:-outputs/robocasa/single_task/vf/indicators_p30.npz}
OUT=${3:-outputs/robocasa/single_task/${MODE}}
# sft baseline: expert-only is enough for BC and ~3x faster. recap: MUST be full finetune
# so the VLM actually consumes the "Advantage: positive/negative" tokens.
# EXPERT override: set EXPERT=full (env) to force a full-finetune sft — needed for a FAIR
# capacity-matched baseline vs recap (else sft=693M expert-only vs recap=4.1B full confounds
# the comparison). EXPERT=expert forces expert-only regardless of mode.
EXPERT_FLAG=""
[ "$MODE" = "sft" ] && EXPERT_FLAG="--train_expert_only"
[ "${EXPERT:-}" = "full" ] && EXPERT_FLAG=""
[ "${EXPERT:-}" = "expert" ] && EXPERT_FLAG="--train_expert_only"
# TASK env selects the single task (per-skill specialist); default CloseFridge.
TASK=${TASK:-CloseFridge}
echo "[sbatch] mode=$MODE task=$TASK ind=$IND out=$OUT expert=${EXPERT_FLAG:-none} node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"

# BASE env overrides the init checkpoint (default lerobot/pi05_base). Paper-faithful
# specialists fine-tune FROM the multi-task pretrained model (e.g. outputs/robocasa/multi_task/sft).
BASE=${BASE:-lerobot/pi05_base}
# ROLLOUTS env (comma-separated repo_ids): specialist RECAP on D_ℓ = task demo + its
# rollouts. Requires the combined-dataset path (--tasks/--rollouts, root=.lerobot); the
# indicators ($IND) MUST be built with the same [demo, rollouts...] repo order.
if [ -n "${ROLLOUTS:-}" ]; then
  DATA_ARGS=(--root .lerobot --tasks "$TASK" --rollouts "$ROLLOUTS")
else
  DATA_ARGS=(--root ".lerobot/pepijn223/robocasa_${TASK}" --repo_id "pepijn223/robocasa_${TASK}")
fi
python recap/scripts/train_pi05_recap_robocasa.py \
  "${DATA_ARGS[@]}" \
  --base_ckpt "$BASE" \
  --mode "$MODE" \
  --indicators "$IND" \
  --episodes_per_task 0 \
  --steps 6000 \
  --batch_size 8 \
  --num_workers 8 \
  --lr 2.5e-5 \
  --log_every 50 \
  --save_every 1500 \
  $EXPERT_FLAG \
  --out "$OUT"

echo "[sbatch] done mode=$MODE out=$OUT"
