#!/usr/bin/env bash
# Minimal push: only what Vast needs to run Affinity.
# Usage: bash tools/push_to_vast_minimal.sh ssh3.vast.ai 16969
set -euo pipefail
HOST="${1:?host}"
PORT="${2:?port}"
KEY="${3:-/mnt/c/Users/matth/.ssh/vast_affinity}"
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
REMOTE="/workspace/magaphragma-connectome"

KEY_USE=/tmp/vast_affinity_push_key
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
SSH="ssh -p $PORT -i $KEY_USE -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes"

echo "Minimal push → root@$HOST:$PORT"
$SSH "root@$HOST" "mkdir -p $REMOTE/tools $REMOTE/experiments/phase6e $REMOTE/local_research_build/phase5c-production $REMOTE/third_party"

rsync -avz -e "$SSH" "$ROOT/tools/" "$REMOTE/../magaphragma-connectome/tools/" --rsync-path="mkdir -p $REMOTE/tools && rsync" 2>/dev/null || \
rsync -avz -e "$SSH" "$ROOT/tools/" "root@$HOST:$REMOTE/tools/"

rsync -avz -e "$SSH" \
  "$ROOT/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001" \
  "$ROOT/experiments/phase6e/AFFINITY-ROI-001" \
  "$ROOT/experiments/phase6e/HYPERDRAIN" \
  "$ROOT/experiments/phase6e/VAST-AFFINITY-BENCH-001" \
  "root@$HOST:$REMOTE/experiments/phase6e/"

rsync -avz -e "$SSH" \
  --include='*.json' --exclude='*' \
  "$ROOT/experiments/phase6e/" "root@$HOST:$REMOTE/experiments/phase6e/"

rsync -avz -e "$SSH" \
  "$ROOT/local_research_build/phase5c-production/chunks.json" \
  "root@$HOST:$REMOTE/local_research_build/phase5c-production/"

rsync -avz -e "$SSH" \
  "$ROOT/third_party/segneuron/Train_and_Inference/" \
  "root@$HOST:$REMOTE/third_party/segneuron/Train_and_Inference/"

$SSH "root@$HOST" "chmod +x $REMOTE/tools/*.sh; ls $REMOTE/tools/run_vast_simple.sh; test -f $REMOTE/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt && echo CKPT_OK || echo CKPT_MISSING"

echo
echo "DONE. On Vast:"
echo "  bash /workspace/magaphragma-connectome/tools/run_vast_simple.sh"
