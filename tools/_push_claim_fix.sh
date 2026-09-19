#!/bin/bash
# Push the claim fix + restart helper files to Vast.
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
R=/workspace/magaphragma-connectome
rsync -avz -e "${SSH[*]}" \
  "$ROOT/tools/s7_chunk_claim.py" \
  "$ROOT/tools/run_vast_simple.sh" \
  "$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py" \
  "root@ssh3.vast.ai:$R/tools/"
"${SSH[@]}" root@ssh3.vast.ai "chmod +x $R/tools/run_vast_simple.sh; echo PUSHED"
