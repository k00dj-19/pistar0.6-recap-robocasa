#!/bin/bash
#SBATCH --job-name=rc_eval
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Closed-loop RoboCasa eval of OUR OWN finetune (factor 6), via the proven eval_cli_wrap
# (lerobot-eval CLI + optional RECAP advantage conditioning). Our policy was trained with the
# dataset's raw camera keys (robot0_agentview_left/right, robot0_eye_in_hand) which the
# RoboCasaEnv emits verbatim -> NO --rename_map needed.
#
# Usage: sbatch slurm/eval_robocasa.sbatch <policy_path> <cond:none|positive> [task] [n_eps] [n_act]
#   none     -> plain eval (SFT baseline, or recap policy without conditioning)
#   positive -> inject " Advantage: positive" into the prompt (RECAP β=1 inference)
set -uo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"

export MUJOCO_GL=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
# GLVND EGL frontend (vendored) + torchcodec NPP runtime (eval env still uses torchcodec for
# the env image pipeline; training used pyav). Same LD_LIBRARY_PATH recipe as the baseline.
GLVND_LIB="$PROJ/glvnd/root/usr/lib/x86_64-linux-gnu"
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$GLVND_LIB:$SP/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

POLICY=${1:?need policy path}
COND=${2:-none}
TASK=${3:-CloseFridge}
NEP=${4:-20}
NACT=${5:-10}
SEED=${SEED:-1000}
[ "$COND" = "none" ] && export ADV_COND="" || export ADV_COND="$COND"
TAG=$(basename "$POLICY")_${TASK}_${COND}_nact${NACT}
OUTDIR=$PROJ/outputs/eval/robocasa_${TAG}_${SLURM_JOB_ID:-local}
mkdir -p "$OUTDIR"
echo "[evalw] policy=$POLICY cond=$COND task=$TASK nep=$NEP nact=$NACT -> $OUTDIR"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# NOTE: run python DIRECTLY (no srun). When this job is submitted from inside another job
# (the watcher worker), srun inherits the parent's SLURM_CPUS_PER_TASK / SLURM_TRES_PER_TASK
# and dies with "cpus-per-task set by two different environment variables" (§5 env leak).
# A 1-GPU single-process eval doesn't need srun; unset the leaked vars for good measure.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPUS_PER_GPU 2>/dev/null || true
python recap/scripts/eval_cli_wrap.py \
  --policy.path="$POLICY" \
  --env.type=robocasa \
  --env.task="$TASK" \
  --eval.batch_size=1 \
  --eval.n_episodes="$NEP" \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.n_action_steps="$NACT" \
  --seed="$SEED" \
  --output_dir="$OUTDIR"; rc=$?
echo "[evalw] DONE $TAG -> $OUTDIR (rc=$rc)"
# done-marker for the output-driven manager (only on success, only for a local policy dir).
if [ "$rc" = "0" ] && [ -d "$POLICY" ]; then
  echo "outdir=$OUTDIR seed=$SEED nep=$NEP" > "${POLICY}/eval_seed${SEED}.done"
fi
