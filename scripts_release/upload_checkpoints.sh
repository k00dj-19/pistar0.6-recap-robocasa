#!/bin/bash
# Upload trained checkpoints to the HuggingFace Hub (public).
#
# Prereqs (you must do these):
#   1) huggingface-cli login            # write token — interactive, secret
#   2) pass your HF username/org as $1
#
# Usage: bash scripts_release/upload_checkpoints.sh <HF_USER> [CKPT_ROOT] [WHICH]
#   WHICH = all (default) | sft | vlmvf | pretrain
# Only model weights + config are uploaded (*.safetensors / *.json); optimizer/resume state
# (resume_state.pt) is excluded. Each π0.5 model is ~16 GB.
set -euo pipefail
HF_USER=${1:?need HF username/org}
ROOT=${2:-outputs/robocasa}
WHICH=${3:-all}

# per-task best VLM-VF iteration (see tutorial.ipynb §4): CF=i1, OD=i1, OC=i2, PnP=i3
declare -A BEST=( [CloseFridge]=recap_vlmvf_iter1 [OpenDrawer]=recap_vlmvf_iter1 \
                  [OpenCabinet]=recap_vlmvf_iter2 [PickPlaceCounterToCabinet]=recap_vlmvf_iter3 )

upload() {  # upload <local_dir> <repo_name>
  local dir name=$2
  dir=$(readlink -f "$1")  # resolve symlinks (PnP iter3 -> recap_vlmvf)
  [ -f "$dir/model.safetensors" ] || { echo "skip (no ckpt): $dir"; return; }
  echo "== $dir -> $HF_USER/$name (~$(du -sh "$dir/model.safetensors" | cut -f1))"
  # `hf` (huggingface_hub CLI; `huggingface-cli` is deprecated). hf upload auto-creates the repo.
  hf upload "$HF_USER/$name" "$dir" . --repo-type model \
    --include "*.safetensors" --include "*.json" --commit-message "upload $name"
}

if [ "$WHICH" = all ] || [ "$WHICH" = pretrain ]; then
  upload "$ROOT/multi_task/sft" "recap-robocasa-pretrain"
fi
for t in CloseFridge OpenDrawer OpenCabinet PickPlaceCounterToCabinet; do
  short=$t; [ "$t" = "PickPlaceCounterToCabinet" ] && short=PnPCounterToCab
  if [ "$WHICH" = all ] || [ "$WHICH" = sft ]; then
    upload "$ROOT/specialist_v2/$t/sft" "recap-robocasa-${short}-sft"
  fi
  if [ "$WHICH" = all ] || [ "$WHICH" = vlmvf ]; then
    upload "$ROOT/specialist_v2/$t/${BEST[$t]}" "recap-robocasa-${short}-vlmvf"
  fi
done
echo "done. Now fill '$HF_USER' into README §5 / tutorial.ipynb HF_USER."
