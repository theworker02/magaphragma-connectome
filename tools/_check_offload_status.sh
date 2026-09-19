#!/bin/bash
echo "=== offload procs ==="
pgrep -af offload_vast_s7 || echo none
echo "=== pidfile ==="
cat ~/s7-affinity-out/vast_offload.pid 2>/dev/null || echo no_pidfile
echo "=== vast ==="
ssh -i /tmp/vast_affinity -p 16969 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@ssh3.vast.ai bash -s <<'REMOTE'
df -h / | head -2
echo PROCS:
pgrep -af run_affinity || true
pgrep -af vast_worker_loop || true
echo CHUNKS:
ls -la /workspace/s7-out/chunks/
echo LOG:
tail -5 /workspace/s7-out/worker.log
REMOTE
echo "=== local shard0 ==="
pgrep -af 'shard 0/2' || echo none
echo "=== offload log tail ==="
tail -20 ~/s7-affinity-out/vast_offload.log
