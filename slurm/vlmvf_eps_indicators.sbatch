#!/bin/bash
#SBATCH --job-name=vlmvf_eps_ind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --qos=own
#SBATCH --time=01:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --error=slurm/logs/%x_%j.log

# ε-sweep indicators for the VLM-VF specialists. ε (positive_fraction) only changes the
# advantage THRESHOLD, not the value function — so we reuse each task's trained vf.pt +
# cached features.npy and just recompute indicators at ε ∈ {0.1, 0.5} (0.3 already done in
# the main run). One job generates all 4 tasks × 2 ε = 8 npzs into <task>/eps_sweep/.
set -uo pipefail
PROJ=/home/nas_main/dongjinkim/pi_06_star
cd "$PROJ"; source "$PROJ/.venv_robocasa/bin/activate"
export HF_HOME="$PROJ/.hf" HF_LEROBOT_HOME="$PROJ/.lerobot" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=recap TOKENIZERS_PARALLELISM=false
SPEC=outputs/robocasa/specialist_v2

for T in PickPlaceCounterToCabinet CloseFridge OpenDrawer OpenCabinet; do
  FEATS=$SPEC/$T/vlmvf/features.npy
  VF=$SPEC/$T/vlmvf_iter3/vf.pt; [ -f "$VF" ] || VF=$SPEC/$T/vlmvf/vf.pt
  ROLLOUTS=$(python -c "import numpy as np; print(' '.join(np.load('$SPEC/$T/vf_iter3/indicators_p30corr.npz',allow_pickle=True)['repo_ids'].tolist()[1:]))")
  OUT=$SPEC/$T/eps_sweep; mkdir -p "$OUT"
  for PF in 0.1 0.5; do
    echo "=== $T pf=$PF (vf=$VF) ==="
    python recap/scripts/compute_indicators_robocasa_multitask.py \
      --root .lerobot --demos pepijn223/robocasa_$T --rollouts $ROLLOUTS \
      --vf "$VF" --features "$FEATS" --out "$OUT" \
      --positive_fraction "$PF" --n_step 50 --demo_positive --tag_suffix corr_vlmvf \
      --device cuda || echo "[warn] $T pf=$PF failed"
  done
done
echo "[done] eps-sweep indicators -> $SPEC/<task>/eps_sweep/indicators_p{10,50}corr_vlmvf.npz"
