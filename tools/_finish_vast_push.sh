#!/bin/bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
R=/workspace/magaphragma-connectome
"${SSH[@]}" root@ssh3.vast.ai "mkdir -p $R/local_research_build/phase5c-production $R/third_party/segneuron/Train_and_Inference $R/experiments/phase6e/AFFINITY-ROI-001/package $R/tools/hyperdrain $R/tools/cloud"
rsync -avz -e "${SSH[*]}" "$ROOT/local_research_build/phase5c-production/chunks.json" "root@ssh3.vast.ai:$R/local_research_build/phase5c-production/"
rsync -avz -e "${SSH[*]}" "$ROOT/third_party/segneuron/Train_and_Inference/" "root@ssh3.vast.ai:$R/third_party/segneuron/Train_and_Inference/"
rsync -avz -e "${SSH[*]}" "$ROOT/tools/hyperdrain/" "root@ssh3.vast.ai:$R/tools/hyperdrain/"
rsync -avz -e "${SSH[*]}" "$ROOT/tools/cloud/" "root@ssh3.vast.ai:$R/tools/cloud/"
rsync -avz -e "${SSH[*]}" \
  "$ROOT/tools/s7_chunk_claim.py" \
  "$ROOT/tools/s7_durable_commit.py" \
  "$ROOT/tools/s7_infer_accelerate.py" \
  "$ROOT/tools/run_vast_simple.sh" \
  "root@ssh3.vast.ai:$R/tools/"
rsync -avz -e "${SSH[*]}" \
  "$ROOT/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json" \
  "root@ssh3.vast.ai:$R/experiments/phase6e/"
rsync -avz -e "${SSH[*]}" "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package/" "root@ssh3.vast.ai:$R/experiments/phase6e/AFFINITY-ROI-001/package/"
"${SSH[@]}" root@ssh3.vast.ai "test -f $R/local_research_build/phase5c-production/chunks.json && test -f $R/third_party/segneuron/Train_and_Inference/model/Mnet.py && echo ALL_READY"
