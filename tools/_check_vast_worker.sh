#!/bin/bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes root@ssh3.vast.ai 'ps aux | grep affinity | grep -v grep; echo ---; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv; echo ---; ls /workspace/s7-out/chunks 2>/dev/null | wc -l'
