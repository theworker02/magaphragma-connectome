# Emergency Affinity fleet fix (Windows OpenSSH). No secrets printed.
$ErrorActionPreference = "Continue"
$Root = "C:\Users\matth\OneDrive\Desktop\magaphragma-connectome"
$Key = "$env:USERPROFILE\.ssh\vast_affinity"
$Remote = "/workspace/magaphragma-connectome"
$LogDir = Join-Path $env:TEMP ("affinity_fix_" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$sshBase = @("-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10")

$Gate = @(
  "$Root\experiments\phase6e\AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
)

# Live affinity-s7 from status + shard map (0 reserved for orig 3060 Ti)
$Targets = @(
  @{H="ssh2.vast.ai"; P=24958; Shard="1/10"; Wid="vast-2080ti"; Label="2080Ti"},
  @{H="ssh1.vast.ai"; P=24958; Shard="2/10"; Wid="vast-3060"; Label="3060"},
  @{H="ssh6.vast.ai"; P=24956; Shard="3/10"; Wid="vast-3080"; Label="3080"},
  @{H="ssh6.vast.ai"; P=24954; Shard="4/10"; Wid="vast-4060ti-b"; Label="4060Ti-b"},
  @{H="ssh6.vast.ai"; P=22706; Shard="5/10"; Wid="vast-5060ti"; Label="5060Ti"},
  @{H="ssh3.vast.ai"; P=22708; Shard="6/10"; Wid="vast-a4000-a"; Label="A4000-a"},
  @{H="ssh6.vast.ai"; P=22712; Shard="7/10"; Wid="vast-a4000-b"; Label="A4000-b"}
)

function Invoke-Ssh([string]$HostName, [int]$Port, [string]$RemoteCmd) {
  & ssh.exe @sshBase -p $Port "root@$HostName" $RemoteCmd
  return $LASTEXITCODE
}

function Invoke-Scp([int]$Port, [string[]]$Files, [string]$Dest) {
  & scp.exe @("-P", "$Port", "-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes") @Files $Dest
  return $LASTEXITCODE
}

function Fix-One($t) {
  $log = Join-Path $LogDir ($t.Label + ".log")
  $sb = {
    param($t, $Root, $Key, $Remote, $Gate, $sshBase, $log)
    function Invoke-Ssh([string]$HostName, [int]$Port, [string]$RemoteCmd) {
      & ssh.exe @sshBase -p $Port "root@$HostName" $RemoteCmd 2>&1 | Out-String
      return $LASTEXITCODE
    }
    $out = New-Object System.Text.StringBuilder
    [void]$out.AppendLine("=== $($t.Label) $($t.H):$($t.P) shard=$($t.Shard) ===")
    $probe = & ssh.exe @sshBase -p $t.P "root@$($t.H)" "echo SSH_OK; nvidia-smi -L | head -1" 2>&1 | Out-String
    [void]$out.AppendLine($probe)
    if ($probe -notmatch "SSH_OK") {
      [void]$out.AppendLine("FAIL_SSH")
      Set-Content -Path $log -Value $out.ToString()
      return 1
    }
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "mkdir -p $Remote/experiments/phase6e $Remote/tools $Remote/third_party/segneuron/Train_and_Inference/model $Remote/local_research_build/phase5c-production $Remote/experiments/phase6e/AFFINITY-ROI-001/package $Remote/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints /tmp/s7-logs /workspace/s7-out/chunks /tmp/s7-accum /tmp/s7-fast-stage" 2>&1 | Out-Null

    $scpArgs = @("-P", "$($t.P)", "-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=40", "-o", "BatchMode=yes")
    $r1 = & scp.exe @scpArgs @Gate "root@$($t.H):$Remote/experiments/phase6e/" 2>&1 | Out-String
    [void]$out.AppendLine("GATE: $r1")
    $tools = @(
      "$Root\tools\run_vast_simple.sh",
      "$Root\tools\run_affinity_fullvol_s7_fast_worker.py",
      "$Root\tools\s7_chunk_claim.py",
      "$Root\tools\s7_durable_commit.py",
      "$Root\tools\s7_infer_accelerate.py"
    )
    $r2 = & scp.exe @scpArgs @tools "root@$($t.H):$Remote/tools/" 2>&1 | Out-String
    [void]$out.AppendLine("TOOLS: $r2")

    # MNet model dir (recursive)
    $r3 = & scp.exe @scpArgs -r "$Root\third_party\segneuron\Train_and_Inference\model" "root@$($t.H):$Remote/third_party/segneuron/Train_and_Inference/" 2>&1 | Out-String
    [void]$out.AppendLine("MNET: $r3")

    # chunks if missing
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "test -f $Remote/local_research_build/phase5c-production/chunks.json" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      & scp.exe @scpArgs "$Root\local_research_build\phase5c-production\chunks.json" "root@$($t.H):$Remote/local_research_build/phase5c-production/" 2>&1 | Out-Null
    }
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "test -f $Remote/experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      & scp.exe @scpArgs -r "$Root\experiments\phase6e\AFFINITY-ROI-001\package\." "root@$($t.H):$Remote/experiments/phase6e/AFFINITY-ROI-001/package/" 2>&1 | Out-Null
    }
    $ckptDst = "$Remote/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints"
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "test -f $ckptDst/checkpoint-step10.pt" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      & scp.exe @scpArgs "$Root\experiments\phase6e\AFFINITY-PARALLEL-FUNNEL-001\C_PATCH_20_64_64\extend\checkpoints\checkpoint-step10.pt" "root@$($t.H):$ckptDst/" 2>&1 | Out-Null
    }

    $remoteScript = @"
set -euo pipefail
test -f $Remote/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json
python3 -c "import json; assert json.load(open('$Remote/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json'))['status']=='APPROVED'; print('S6_OK')"
test -f $Remote/third_party/segneuron/Train_and_Inference/model/Mnet.py && echo MNET_OK
test -f $Remote/tools/s7_infer_accelerate.py && echo ACCEL_OK
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-logs /tmp/s7-accum /tmp/s7-fast-stage
: > /tmp/s7-logs/worker.log
BATCH=12
MEM=`$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 16000)
if [ -n "`${MEM:-}" ] && [ "`$MEM" -lt 10000 ]; then BATCH=8; fi
cd $Remote
nohup env S7_SHARD=$($t.Shard) S7_WORKER_ID=$($t.Wid) S7_TILE_BATCH=`$BATCH bash -c 'while true; do bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1; echo respawn >>/tmp/s7-logs/worker.log; sleep 20; done' >/tmp/s7-logs/loop.log 2>&1 &
echo STARTED BATCH=`$BATCH SHARD=$($t.Shard)
sleep 16
echo '--- log ---'
head -n 100 /tmp/s7-logs/worker.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
if grep -q "FileNotFoundError.*AFFINITY_S6" /tmp/s7-logs/worker.log; then echo FAIL_S6; exit 3; fi
if grep -q "No module named 'model'" /tmp/s7-logs/worker.log; then echo FAIL_MODEL; exit 4; fi
if grep -qE 'FAST infer|tiles_per_sec|infer_backend|queue pending' /tmp/s7-logs/worker.log; then echo LOOKS_ALIVE; fi
"@
    $rr = & ssh.exe @sshBase -p $t.P "root@$($t.H)" $remoteScript 2>&1 | Out-String
    [void]$out.AppendLine($rr)
    $rc = 0
    if ($rr -match "FAIL_S6|FAIL_MODEL|FAIL_SSH") { $rc = 1 }
    elseif ($rr -match "LOOKS_ALIVE|tiles_per_sec|FAST infer") { $rc = 0 }
    elseif ($rr -match "Traceback|Error") { $rc = 1 }
    Set-Content -Path $log -Value $out.ToString()
    return $rc
  }
  Start-Job -ScriptBlock $sb -ArgumentList $t, $Root, $Key, $Remote, $Gate, $sshBase, $log | Out-Null
}

Write-Host "LOGDIR=$LogDir"
foreach ($t in $Targets) { Fix-One $t }

# Confirm original 3060 Ti — do not destroy; retarget only if needed
$origJob = Start-Job -ScriptBlock {
  param($Key, $sshBase, $log)
  $r = & ssh.exe @sshBase -p 28225 "root@92.190.14.191" @'
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
PID=$(pgrep -f run_affinity_fullvol_s7_fast_worker | head -1)
SH=""
if [ -n "$PID" ]; then SH=$(tr "\0" "\n" < /proc/$PID/environ 2>/dev/null | grep "^S7_SHARD=" || true); fi
echo "util=$UTIL shard_env=$SH"
tail -10 /tmp/s7-logs/worker.log 2>/dev/null || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
# If util high and already 0/10, leave alone
if [ "${UTIL:-0}" -gt 5 ] && echo "$SH" | grep -q "0/10"; then echo LEAVE_ORIG_ALONE; exit 0; fi
# If on 1/2 or idle wrong shard, retarget to 0/10 without wiping disk
if [ "${UTIL:-0}" -le 5 ] || echo "$SH" | grep -qE "1/2"; then
  echo RETARGET_0_10
  test -f /workspace/magaphragma-connectome/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json || exit 5
  pkill -f run_affinity_fullvol_s7_fast_worker 2>/dev/null || true
  pkill -f run_vast_simple.sh 2>/dev/null || true
  sleep 2
  mkdir -p /tmp/s7-logs
  cd /workspace/magaphragma-connectome
  nohup env S7_SHARD=0/10 S7_WORKER_ID=vast-3060ti S7_TILE_BATCH=8 bash -c "while true; do bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1; sleep 20; done" >/tmp/s7-logs/loop.log 2>&1 &
  sleep 10
  tail -20 /tmp/s7-logs/worker.log
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
fi
'@ 2>&1 | Out-String
  Set-Content -Path $log -Value $r
} -ArgumentList $Key, $sshBase, (Join-Path $LogDir "3060Ti-orig.log")

Write-Host "Waiting for jobs..."
Get-Job | Wait-Job -Timeout 420 | Out-Null
Get-Job | Receive-Job 2>$null | Out-Null
Get-Job | Remove-Job -Force

Write-Host "==== SUMMARIES ===="
Get-ChildItem $LogDir -Filter *.log | ForEach-Object {
  Write-Host ("----- " + $_.Name + " -----")
  Get-Content $_.FullName -Tail 40
  Write-Host ""
}
Write-Host "DONE logs=$LogDir"
