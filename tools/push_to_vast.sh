#!/usr/bin/env bash
# ONE command: copy Affinity code + checkpoint to Vast.
# Usage:
#   bash tools/push_to_vast.sh ssh3.vast.ai 16969
# Optional 3rd arg: path to SSH private key (default: tries vast_affinity keys)
set -euo pipefail
HOST="${1:?Vast SSH host e.g. ssh3.vast.ai}"
PORT="${2:?Vast SSH port}"
KEY="${3:-}"
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
REMOTE="/workspace/magaphragma-connectome"

if [[ -z "$KEY" ]]; then
  for candidate in \
    "$HOME/.ssh/vast_affinity" \
    /mnt/c/Users/matth/.ssh/vast_affinity \
    "$HOME/.ssh/id_ed25519" \
    /mnt/c/Users/matth/.ssh/id_ed25519
  do
    if [[ -f "$candidate" ]]; then
      KEY="$candidate"
      break
    fi
  done
fi
if [[ -z "${KEY}" || ! -f "$KEY" ]]; then
  echo "No SSH private key found. Pass it as 3rd arg."
  exit 2
fi
# WSL cannot use Windows-ACL keys reliably — copy to /tmp with 600
KEY_USE="/tmp/vast_affinity_push_key"
cp "$KEY" "$KEY_USE"
chmod 600 "$KEY_USE"

SSH_CMD="ssh -p $PORT -i $KEY_USE -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes"

echo "Pushing to root@$HOST:$PORT:$REMOTE (key=$KEY) ..."
$SSH_CMD "root@$HOST" "mkdir -p $REMOTE"
rsync -avz -e "$SSH_CMD" \
  --exclude '.venv' --exclude '.venv-reviewer' --exclude '**/__pycache__' \
  --exclude 'experiments/phase6e/AFFINITY-FULLVOL-S7-001/chunks' \
  --exclude 'local_research_build/phase6e/independent_model_candidates' \
  --exclude '.git' \
  "$ROOT/" "root@$HOST:$REMOTE/"

echo "Done. On Vast run:"
echo "  bash /workspace/magaphragma-connectome/tools/run_vast_simple.sh"
