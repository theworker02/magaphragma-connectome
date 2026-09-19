#!/bin/bash
set -euo pipefail
echo "=== chunk sizes ==="
du -sh /workspace/s7-out/chunks/MV-CHUNK-* 2>/dev/null | sort -h | tail -40 || true
echo "=== per-chunk inventory ==="
for d in /workspace/s7-out/chunks/MV-CHUNK-*; do
  n=$(ls -A "$d" 2>/dev/null | wc -l)
  sz=$(du -sk "$d" | awk '{print $1}')
  has_r=0; [[ -f "$d/receipt.json" ]] && has_r=1
  has_n=0; [[ -f "$d/affinities_core_czyx.npy" ]] && has_n=1
  echo "$n files ${sz}k receipt=$has_r npy=$has_n $(basename "$d")"
done
echo "=== clear caches ==="
rm -rf /root/.cache/pip /tmp/s7-accum /tmp/s7-fast-stage 2>/dev/null || true
mkdir -p /tmp/s7-accum /tmp/s7-fast-stage
n_rm=0
for d in /workspace/s7-out/chunks/MV-CHUNK-*; do
  [[ -d "$d" ]] || continue
  if [[ ! -f "$d/receipt.json" ]]; then
    asz=$(stat -c%s "$d/affinities_core_czyx.npy" 2>/dev/null || echo 0)
    if [[ "$asz" -lt 1000000 ]]; then
      rm -rf "$d"
      n_rm=$((n_rm+1))
    fi
  fi
done
echo "removed_incomplete=$n_rm"
# Drop bulky local copies not needed for inference if present
# Keep checkpoint; trim unrelated datasets if >2G and not required
if [[ -d /workspace/magaphragma-connectome/datasets ]]; then
  du -sh /workspace/magaphragma-connectome/datasets/* 2>/dev/null | sort -h | tail -10 || true
fi
df -h /
echo "=== nvidia reset attempt ==="
nvidia-smi --gpu-reset 2>&1 || true
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv || true
