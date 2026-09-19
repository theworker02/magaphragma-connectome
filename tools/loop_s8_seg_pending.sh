#!/usr/bin/env bash
set -euo pipefail
PY=/opt/venvs/unlearning-rocm/bin/python
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
LOG=$ROOT/experiments/phase6e/AFFINITY-FULLVOL-S7-001/s8_seg_worker.log
"$PY" -c 'import elf.segmentation.watershed, nifty; print("elf_nifty_ok")'
exec "$PY" -u "$ROOT/tools/run_s7_seg_pending_worker.py" --max-chunks 4 --loop >>"$LOG" 2>&1
