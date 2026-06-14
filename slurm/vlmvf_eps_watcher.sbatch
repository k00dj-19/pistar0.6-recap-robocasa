#!/bin/bash
#SBATCH --job-name=vlmvf_eps_watch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=own
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Persistent watcher for the VLM-VF ε-sweep (8 cells = 4 tasks × ε{0.1,0.5}). Loops the
# idempotent eps-manager every INTERVAL: advances recap→eval, auto-resubmits preempted jobs.
# CPU-only + own QOS so the watcher is never preempted. Exits when all 8 cells eval-complete.
set -uo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"; source "$PROJ/.venv_robocasa/bin/activate"
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
INTERVAL=${INTERVAL:-180}
SPEC=outputs/robocasa/specialist_v2
all_done() { local T e; for T in PickPlaceCounterToCabinet CloseFridge OpenDrawer OpenCabinet; do
  for e in 10 50; do [ -f "$SPEC/$T/recap_vlmvf_eps${e}/eval_seed5000.done" ] || return 1; done; done; return 0; }

echo "[epswatch] start $(date -u) interval=${INTERVAL}s node=$(hostname)"
for i in $(seq 1 480); do
  echo "----- tick $i $(date -u +%H:%M:%S) -----"
  bash slurm/vlmvf_eps_manage.sh 2>&1 | sed 's/^/[eps] /' || echo "[epswatch] manager errored (continuing)"
  if all_done; then echo "[epswatch] all 8 eps cells done — exiting"; break; fi
  sleep "$INTERVAL"
done
echo "[epswatch] done $(date -u)"
