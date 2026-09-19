# Hybrid Affinity fleet: local RX 7800 XT + Vast RTX 3060 Ti

Shared queue coordinator runs on the **local** machine that owns
`experiments/phase6e/AFFINITY-FULLVOL-S7-001/`.

## Why dual-claim is impossible

Claims create `claims/<chunk_id>.claim.json` with `os.O_EXCL` on the controller
filesystem. Only one create can succeed. Vast workers call HTTP `/claim` or
`/next`, which invoke the same `try_claim_local`. Local workers call
`try_claim_local` directly on the same directory. Second claimer always loses.

Complete requires durable `affinities_core_czyx.npy` on the controller before
status becomes `AFFINITY_DONE_SEG_PENDING`.

## Networking

Vast must reach `http://<CONTROLLER>:8787`.

Options (pick one):

1. **Tailscale / ZeroTier** on both hosts (recommended)
2. **SSH reverse tunnel** from Vast → local:  
   `ssh -R 8787:127.0.0.1:8787 user@vast` (run on local)  
   then Vast uses `S7_CLAIM_HTTP=http://127.0.0.1:8787`
3. Port-forward / public IP (firewall carefully)

Artifact durability uses `S7_DURABLE_MODE=http` (upload npy to coordinator)
or `rsync` when SSH to the controller is available.

## Exact commands

### LOCAL (WSL) — RX 7800 XT

```bash
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
bash tools/start_local_affinity_fleet.sh
```

Or step-by-step:

```bash
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
PY=/opt/venvs/unlearning-rocm/bin/python

# Shared claim coordinator (keep running)
$PY -u tools/cloud/claim_http.py

# Other WSL terminal — local free worker
export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=16
export S7_INFER_BACKEND=eager
$PY -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 64 --batch-size 16 \
  --contract FAST_003 --infer-backend eager

# Status anytime
$PY tools/affinity_fleet.py status --claim-http http://127.0.0.1:8787

# Record Vast $/hour from the Vast UI (replace values)
$PY tools/record_vast_instance_cost.py \
  --instance-id YOUR_INSTANCE_ID \
  --gpu "NVIDIA GeForce RTX 3060 Ti" \
  --dph 0.00 \
  --worker-id vast-rtx3060ti-01
```

Expose port 8787 to Vast (Tailscale IP recommended), then set on Vast:
`S7_CLAIM_HTTP=http://CONTROLLER_IP:8787`

### Deploy repo + checkpoint TO Vast (from WSL)

```bash
# Set these from the Vast SSH/Jupyter connection panel:
export VAST_HOST=...
export VAST_PORT=...

rsync -avz -e "ssh -p $VAST_PORT" \
  --exclude '.venv' --exclude '.venv-reviewer' --exclude '**/__pycache__' \
  --exclude 'experiments/phase6e/AFFINITY-FULLVOL-S7-001/chunks' \
  /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/ \
  root@$VAST_HOST:/workspace/magaphragma-connectome/

# Checkpoint (required; not always in git)
rsync -avz -e "ssh -p $VAST_PORT" \
  /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt \
  root@$VAST_HOST:/workspace/magaphragma-connectome/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt
```

### VAST RTX 3060 Ti (paste in Jupyter terminal)

```bash
export S7_CLAIM_HTTP=http://CONTROLLER_IP:8787   # <-- your coordinator
export S7_WORKER_ID=vast-rtx3060ti-01
export S7_VAST_DPH=0.00                          # <-- actual $/h from Vast UI
export S7_VAST_INSTANCE_ID=YOUR_INSTANCE_ID
export S7_DURABLE_MODE=http
export S7_CLAIM_BACKEND=http
export AFFINITY_ROOT=/workspace/magaphragma-connectome

cd /workspace/magaphragma-connectome
bash tools/bootstrap_vast_affinity.sh
```

Bootstrap will: verify GPU → verify checkpoint SHA256 → hit `/health` → run
`qualify_nvidia_affinity.py` → only then `--claim --loop`.
