#!/bin/bash
# Idempotent output-driven manager for the VLM-VF ε-sweep (4 tasks × ε∈{0.1,0.5}, iter3 data).
# Reuses each task's vf.pt + cached features (ε only changes indicator threshold). Submits
# recap (4-GPU, own/extra/share) then eval (1-GPU, extra/share) per cell; resume-capable.
# Indicators are produced by slurm/vlmvf_eps_indicators.sbatch into <task>/eps_sweep/.
set -uo pipefail
cd /home/nas_main/dongjinkim/pi_06_star
PY=.venv_robocasa/bin/python
BASE=outputs/robocasa/multi_task/sft
SPEC=outputs/robocasa/specialist_v2
SEED=5000
TASKS="PickPlaceCounterToCabinet CloseFridge OpenDrawer OpenCabinet"
EPS="10 50"
QCNT_FILE=/tmp/vlmvf_eps_qcnt_$USER
DRYRUN=${DRYRUN:-0}

queued() { squeue -u "$USER" -h -o "%j|%T" 2>/dev/null | grep -qE "^$1\|(PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED)$"; }
next_qos() { local n=0; [ -f "$QCNT_FILE" ] && n=$(cat "$QCNT_FILE" 2>/dev/null||echo 0); echo $(((n+1)%3)) > "$QCNT_FILE"
  case $((n%3)) in 0) echo "--qos=own";; 1) echo "--qos=extra";; 2) echo "-A share --qos=share";; esac; }
light_qos() { local n=0; [ -f "$QCNT_FILE.l" ] && n=$(cat "$QCNT_FILE.l" 2>/dev/null||echo 0); echo $(((n+1)%2)) > "$QCNT_FILE.l"
  case $((n%2)) in 0) echo "--qos=extra";; 1) echo "-A share --qos=share";; esac; }
sub() { local name="$1"; shift; local env="$1"; shift; local q
  case "$name" in ev_*) q=$(light_qos);; *) q=$(next_qos);; esac
  if [ "$DRYRUN" = "1" ]; then echo "WOULD: $env sbatch $q -J $name $*"; return; fi
  eval "$env sbatch $q -J $name $*" && echo "[eps] submitted $name ($q)"; }
rollouts_csv() { $PY -c "import numpy as np; print(','.join(np.load('$SPEC/$1/vf_iter3/indicators_p30corr.npz',allow_pickle=True)['repo_ids'].tolist()[1:]))" 2>/dev/null; }

for T in $TASKS; do
  RCSV=$(rollouts_csv "$T"); [ -z "$RCSV" ] && continue
  for e in $EPS; do
    IND=$SPEC/$T/eps_sweep/indicators_p${e}corr_vlmvf.npz
    RECAP=$SPEC/$T/recap_vlmvf_eps${e}
    [ -f "$IND" ] || { echo "[eps] $T eps$e: indicators not ready"; continue; }
    [ -f "$RECAP/eval_seed${SEED}.done" ] && continue
    if [ -f "$RECAP/TRAIN_DONE" ]; then
      queued "ev_eps${e}_$T" || sub "ev_eps${e}_$T" "SEED=$SEED" slurm/eval_robocasa.sbatch "$RECAP" positive "$T" 50 10
      continue
    fi
    queued "rc_eps${e}_$T" || sub "rc_eps${e}_$T" \
      "ROLLOUTS='$RCSV' TASK='$T' BASE='$BASE' EXPERT=full" \
      slurm/finetune_robocasa_4gpu.sbatch recap "$IND" "$RECAP"
  done
done
echo "[eps] pass complete $(date -u +%H:%M:%S)"
