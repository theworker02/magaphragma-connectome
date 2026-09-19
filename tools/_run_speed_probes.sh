#!/bin/bash
kill -STOP 303 2>/dev/null || true
sleep 1
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
PY=/opt/venvs/unlearning-rocm/bin/python
echo "=== MIOPEN env variants ==="
run_one() {
  local label="$1"; shift
  echo "RUN $label"
  env "$@" PROBE_LABEL="$label" HYPERDRAIN_SKIP_VRAM_SEARCH=1 "$PY" -u tools/_probe_env_tps.py || true
}
run_one baseline
run_one miopen_fast MIOPEN_FIND_MODE=FAST
run_one miopen_normal MIOPEN_FIND_MODE=NORMAL
run_one hipblaslt TORCH_BLAS_PREFER_HIPBLASLT=1
echo "=== compile safe ==="
"$PY" -u tools/_probe_compile_safe.py || true
kill -CONT 303 2>/dev/null || true
echo DONE
