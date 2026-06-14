#!/bin/bash
# Idempotent, output-driven manager for the VLM-VF K=3 x 4-task experiment matrix.
# Each run brings the job set toward completion: it inspects DONE-markers + the live queue,
# then submits ONLY the missing, not-yet-queued stages. Safe to run repeatedly (this is what
# the watcher loops). Spreads GPU jobs across own/extra/share for maximum concurrency; all
# stages are resume-capable (extract shards, RECAP checkpoint) so preemption is non-fatal.
#
# Stage markers (per task T, iter k):
#   extract  -> $SPEC/$T/vlmvf/features.npy
#   vf+ind   -> $SPEC/$T/vlmvf_iter$k/indicators_p30corr_vlmvf.npz
#   recap    -> $SPEC/$T/recap_vlmvf_iter$k/TRAIN_DONE
#   eval     -> $SPEC/$T/recap_vlmvf_iter$k/eval_seed$SEED.done
#
# Usage: bash slurm/vlmvf_manage.sh            (submit)
#        DRYRUN=1 bash slurm/vlmvf_manage.sh   (print what it WOULD submit)
set -uo pipefail
cd /home/nas_main/dongjinkim/pi_06_star
PY=.venv_robocasa/bin/python
BASE=outputs/robocasa/multi_task/sft
SPEC=outputs/robocasa/specialist_v2
SEED=5000
TASKS="PickPlaceCounterToCabinet CloseFridge OpenDrawer OpenCabinet"
QCNT_FILE=/tmp/vlmvf_qcnt_$USER
DRYRUN=${DRYRUN:-0}

queued() {  # 0 if a non-terminal job with EXACT name $1 exists
  squeue -u "$USER" -h -o "%j|%T" 2>/dev/null \
    | grep -qE "^$1\|(PENDING|RUNNING|CONFIGURING|COMPLETING|RESV_DEL_HOLD|REQUEUED)$"
}
next_qos() {  # 4-GPU jobs (extract/recap): rotate own -> extra -> share
  local n=0; [ -f "$QCNT_FILE" ] && n=$(cat "$QCNT_FILE" 2>/dev/null || echo 0)
  echo $(( (n+1) % 3 )) > "$QCNT_FILE"
  case $((n % 3)) in
    0) echo "--qos=own" ;;
    1) echo "--qos=extra" ;;
    2) echo "-A share --qos=share" ;;
  esac
}
light_qos() {  # 1-GPU short jobs (vf/eval): extra/share ONLY — never own, else they starve
  local n=0; [ -f "$QCNT_FILE.light" ] && n=$(cat "$QCNT_FILE.light" 2>/dev/null || echo 0)
  echo $(( (n+1) % 2 )) > "$QCNT_FILE.light"
  case $((n % 2)) in 0) echo "--qos=extra" ;; 1) echo "-A share --qos=share" ;; esac
}
sub() {  # sub "<jobname>" "<env assignments>" sbatch-args...   (honors DRYRUN)
  local name="$1"; shift; local env="$1"; shift
  local q
  case "$name" in
    ev_*|vf_*) q=$(light_qos) ;;   # 1-GPU short -> extra/share (avoid own 4-GPU starvation)
    *)         q=$(next_qos) ;;    # 4-GPU extract/recap -> own/extra/share
  esac
  if [ "$DRYRUN" = "1" ]; then echo "WOULD: $env sbatch $q -J $name $*"; return; fi
  eval "$env sbatch $q -J $name $*" && echo "[mgr] submitted $name ($q)"
}
rollouts_for() {  # full rollout list (space) from the task's corrections npz (npz order)
  $PY -c "import numpy as np; print(' '.join(np.load('$SPEC/$1/vf_iter3/indicators_p30corr.npz',allow_pickle=True)['repo_ids'].tolist()[1:]))" 2>/dev/null
}
prefix_iter() {  # $1=full rollouts(space) $2=iter -> cumulative prefix subset
  local R=($1); local k=$2; local out=(); local r
  for r in "${R[@]}"; do
    if [ "$k" = "1" ]; then [[ "$r" == *iter2* || "$r" == *sp* ]] && break; fi
    if [ "$k" = "2" ]; then [[ "$r" == *sp* ]] && break; fi
    out+=("$r")
  done
  echo "${out[@]}"
}

for T in $TASKS; do
  FULL=$(rollouts_for "$T")
  [ -z "$FULL" ] && { echo "[mgr] WARN no rollouts for $T (missing corrections npz) — skip"; continue; }
  VLMVF=$SPEC/$T/vlmvf
  FEATS=$VLMVF/features.npy

  # ---- stage 0: per-task feature extraction (shared by all iters) ----
  if [ ! -f "$FEATS" ]; then
    queued "ex_$T" || sub "ex_$T" "ROLLOUTS='$FULL' TASK='$T'" slurm/extract_vlmvf_robocasa.sbatch
    continue  # nothing downstream until features.npy exists
  fi

  for k in 1 2 3; do
    RSUB=$(prefix_iter "$FULL" "$k"); RSUB_CSV=$(echo "$RSUB" | tr ' ' ',')
    INDDIR=$SPEC/$T/vlmvf_iter$k
    RECAP=$SPEC/$T/recap_vlmvf_iter$k
    IND=$INDDIR/indicators_p30corr_vlmvf.npz

    [ -f "$RECAP/eval_seed${SEED}.done" ] && continue           # iter fully done
    if [ -f "$RECAP/TRAIN_DONE" ]; then                          # recap done -> eval
      queued "ev_${T}_$k" || sub "ev_${T}_$k" "SEED=$SEED" \
        slurm/eval_robocasa.sbatch "$RECAP" positive "$T" 50 10
      continue
    fi
    if [ -f "$IND" ]; then                                       # indicators done -> recap
      queued "rc_${T}_$k" || sub "rc_${T}_$k" \
        "ROLLOUTS='$RSUB_CSV' TASK='$T' BASE='$BASE' EXPERT=full" \
        slurm/finetune_robocasa_4gpu.sbatch recap "$IND" "$RECAP"
      continue
    fi
    # features exist -> VF + indicators (full features auto-sliced to this iter's prefix)
    queued "vf_${T}_$k" || sub "vf_${T}_$k" \
      "ROLLOUTS='$RSUB' TASK='$T' OUT='$INDDIR' FEATS='$FEATS'" \
      slurm/vlmvf_train_indicators.sbatch
  done
done
echo "[mgr] pass complete $(date -u +%H:%M:%S)"
