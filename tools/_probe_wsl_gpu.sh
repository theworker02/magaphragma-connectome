#!/usr/bin/env bash
set -euo pipefail
echo DISTRO_OK
uname -a
echo --- modules ---
lsmod | grep -E 'amdgpu|amdkfd' || echo NO_AMDGPU
echo --- devices ---
ls -la /dev/kfd /dev/dri 2>&1 | head -20 || true
echo --- rocm ---
command -v rocm-smi || true
if [[ -x /opt/rocm/bin/rocm-smi ]]; then
  /opt/rocm/bin/rocm-smi 2>&1 | head -30
else
  echo no_rocm_smi
fi
echo --- torch ---
if [[ -x /opt/venvs/unlearning-rocm/bin/python ]]; then
  /opt/venvs/unlearning-rocm/bin/python - <<'PY'
import torch
print("cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
else
  echo no_unlearning_rocm_venv
fi
echo --- who ---
id
echo "HOME=$HOME"
ls /home 2>/dev/null || true
