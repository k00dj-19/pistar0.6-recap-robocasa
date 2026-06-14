#!/bin/bash
#SBATCH --job-name=vlmvf_watch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=own
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# Persistent watcher for the VLM-VF K=3 x 4-task matrix. Every INTERVAL it runs the idempotent
# manager, which (a) advances pipelines (submits the next stage once a marker appears) and
# (b) AUTO-RESUBMITS any stage whose output marker is missing and which has no live job —
# i.e. preempted/cancelled extra/share jobs come back automatically (resume-capable, so they
# continue from their last checkpoint/shard). CPU-only (gres=gpu:0) + own QOS so the watcher
# itself is never preempted. Exits on its own once every iter is eval-complete.
set -uo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"
source "$PROJ/.venv_robocasa/bin/activate"
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

INTERVAL=${INTERVAL:-180}
TASKS="PickPlaceCounterToCabinet CloseFridge OpenDrawer OpenCabinet"
SPEC=outputs/robocasa/specialist_v2

all_done() {  # 0 only when every task/iter has its eval marker
  local T k
  for T in $TASKS; do for k in 1 2 3; do
    [ -f "$SPEC/$T/recap_vlmvf_iter$k/eval_seed5000.done" ] || return 1
  done; done
  return 0
}

echo "[watch] start $(date -u) interval=${INTERVAL}s node=$(hostname)"
for i in $(seq 1 960); do   # 960 * 180s = 48h cap
  echo "----- tick $i $(date -u +%H:%M:%S) -----"
  bash slurm/vlmvf_manage.sh 2>&1 | sed 's/^/[mgr] /' || echo "[watch] manager pass errored (continuing)"
  if all_done; then echo "[watch] ALL 12 iters eval-complete — exiting"; break; fi
  sleep "$INTERVAL"
done
echo "[watch] done $(date -u)"
