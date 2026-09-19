#!/usr/bin/env bash
set -euo pipefail
PY=/opt/venvs/unlearning-rocm/bin/python
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
pgrep -af run_affinity_fullvol_s7_fast_worker || echo stopped
"$PY" -c 'import torch_migraphx; print("migraphx_ok", torch_migraphx.__file__)' || echo migraphx_import_failed
"$PY" "$ROOT/tools/bench_s7_infer_backends.py" --batch 16 --steps 48 --warmup 6 --backends eager,compile,migraphx
