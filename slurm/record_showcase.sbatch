#!/bin/bash
#SBATCH --job-name=rc_show
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Record per-episode labeled rollout videos for the tutorial gallery.
# Usage: sbatch slurm/record_showcase.sbatch <ckpt_dir> <task> <cond:none|positive> <out_dir> [n_eps]
set -uo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"; source "$PROJ/.venv_robocasa/bin/activate"
export MUJOCO_GL=egl CMAKE_POLICY_VERSION_MINIMUM=3.5 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=recap
GLVND_LIB="$PROJ/glvnd/root/usr/lib/x86_64-linux-gnu"
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$GLVND_LIB:$SP/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
# avoid srun env-leak if launched from inside another job (§5)
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPUS_PER_GPU 2>/dev/null || true

CKPT=${1:?need ckpt dir}; TASK=${2:?need task}; COND=${3:-none}; OUT=${4:?need out dir}; NEP=${5:-10}
echo "[showcase] ckpt=$CKPT task=$TASK cond=$COND out=$OUT nep=$NEP node=$(hostname)"
python recap/scripts/record_showcase.py \
  --ckpt "$CKPT" --task "$TASK" --condition "$COND" \
  --n_episodes "$NEP" --seed 5000 --n_action_steps 10 \
  --camera robot0_agentview_left --out "$OUT"
echo "[showcase] done -> $OUT"
