# Push restart script + run on hosts that already got gate files. Attach key for A4000-a.
$ErrorActionPreference = "Continue"
$Root = "C:\Users\matth\OneDrive\Desktop\magaphragma-connectome"
$Key = "$env:USERPROFILE\.ssh\vast_affinity"
$Remote = "/workspace/magaphragma-connectome"
$LogDir = Join-Path $env:TEMP ("affinity_restart_" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$sshBase = @("-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes")
$RestartLocal = "$Root\tools\_remote_restart_affinity.sh"

# Attach SSH key for permission-denied hosts
Write-Host "=== attach vast_affinity pubkey to instances ==="
Push-Location $Root
python tools\_attach_fleet_ssh.py 2>&1 | Select-Object -Last 40
Pop-Location

$Targets = @(
  @{H="ssh2.vast.ai"; P=24958; Shard="1/10"; Wid="vast-2080ti"; Label="2080Ti"},
  @{H="ssh1.vast.ai"; P=24958; Shard="2/10"; Wid="vast-3060"; Label="3060"},
  @{H="ssh6.vast.ai"; P=24956; Shard="3/10"; Wid="vast-3080"; Label="3080"},
  @{H="ssh6.vast.ai"; P=24954; Shard="4/10"; Wid="vast-4060ti-b"; Label="4060Ti-b"},
  @{H="ssh6.vast.ai"; P=22706; Shard="5/10"; Wid="vast-5060ti"; Label="5060Ti"},
  @{H="ssh3.vast.ai"; P=22708; Shard="6/10"; Wid="vast-a4000-a"; Label="A4000-a"},
  @{H="ssh6.vast.ai"; P=22712; Shard="7/10"; Wid="vast-a4000-b"; Label="A4000-b"}
)

$Gate = @(
  "$Root\experiments\phase6e\AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json",
  "$Root\experiments\phase6e\AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
)

$jobs = @()
foreach ($t in $Targets) {
  $jobs += Start-Job -ScriptBlock {
    param($t, $Root, $Key, $Remote, $Gate, $sshBase, $RestartLocal, $LogDir)
    $log = Join-Path $LogDir ($t.Label + ".log")
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("=== $($t.Label) $($t.H):$($t.P) $($t.Shard) ===")
    $scp = @("-P", "$($t.P)", "-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=40", "-o", "BatchMode=yes")

    # Ensure dirs + push gate/tools/mnet/restart again (idempotent)
    $mk = & ssh.exe @sshBase -p $t.P "root@$($t.H)" "mkdir -p $Remote/experiments/phase6e $Remote/tools $Remote/third_party/segneuron/Train_and_Inference /tmp/s7-logs" 2>&1 | Out-String
    [void]$sb.AppendLine($mk)
    if ($mk -match "Permission denied|Connection closed|timed out") {
      Set-Content $log $sb.ToString(); return
    }
    & scp.exe @scp @Gate "root@$($t.H):$Remote/experiments/phase6e/" 2>&1 | Out-Null
    $tools = @(
      "$Root\tools\run_vast_simple.sh",
      "$Root\tools\run_affinity_fullvol_s7_fast_worker.py",
      "$Root\tools\s7_chunk_claim.py",
      "$Root\tools\s7_durable_commit.py",
      "$Root\tools\s7_infer_accelerate.py",
      $RestartLocal
    )
    & scp.exe @scp @tools "root@$($t.H):$Remote/tools/" 2>&1 | Out-Null
    & scp.exe @scp -r "$Root\third_party\segneuron\Train_and_Inference\model" "root@$($t.H):$Remote/third_party/segneuron/Train_and_Inference/" 2>&1 | Out-Null

    # chunks/roi/ckpt
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "test -f $Remote/local_research_build/phase5c-production/chunks.json || mkdir -p $Remote/local_research_build/phase5c-production" 2>&1 | Out-Null
    & scp.exe @scp "$Root\local_research_build\phase5c-production\chunks.json" "root@$($t.H):$Remote/local_research_build/phase5c-production/" 2>&1 | Out-Null
    & ssh.exe @sshBase -p $t.P "root@$($t.H)" "mkdir -p $Remote/experiments/phase6e/AFFINITY-ROI-001/package $Remote/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints" 2>&1 | Out-Null
    & scp.exe @scp "$Root\experiments\phase6e\AFFINITY-ROI-001\package\ROI_PACKAGE.json" "root@$($t.H):$Remote/experiments/phase6e/AFFINITY-ROI-001/package/" 2>&1 | Out-Null
    & scp.exe @scp "$Root\experiments\phase6e\AFFINITY-PARALLEL-FUNNEL-001\C_PATCH_20_64_64\extend\checkpoints\checkpoint-step10.pt" "root@$($t.H):$Remote/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/" 2>&1 | Out-Null

    $run = & ssh.exe @sshBase -p $t.P "root@$($t.H)" "sed -i 's/\r$//' $Remote/tools/_remote_restart_affinity.sh; chmod +x $Remote/tools/_remote_restart_affinity.sh; bash $Remote/tools/_remote_restart_affinity.sh $($t.Shard) $($t.Wid)" 2>&1 | Out-String
    [void]$sb.AppendLine($run)
    Set-Content $log $sb.ToString()
  } -ArgumentList $t, $Root, $Key, $Remote, $Gate, $sshBase, $RestartLocal, $LogDir
}

# Orig 3060 Ti confirm / soft retarget
$jobs += Start-Job -ScriptBlock {
  param($Key, $sshBase, $LogDir, $Root)
  $log = Join-Path $LogDir "3060Ti-orig.log"
  $scp = @("-P", "28225", "-i", $Key, "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=35", "-o", "BatchMode=yes")
  $out = New-Object System.Text.StringBuilder
  # push S6 quietly in case missing, do not kill if already inferring 0/10
  $probe = & ssh.exe @sshBase -p 28225 "root@92.190.14.191" "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -af run_affinity_fullvol | head -2; tr '\0' '\n' < /proc/\$(pgrep -f run_affinity_fullvol_s7_fast_worker | head -1)/environ 2>/dev/null | grep S7_SHARD || true; tail -5 /tmp/s7-logs/worker.log 2>/dev/null" 2>&1 | Out-String
  [void]$out.AppendLine($probe)
  if ($probe -match "timed out|Permission denied|Connection") {
    Set-Content $log $out.ToString(); return
  }
  & ssh.exe @sshBase -p 28225 "root@92.190.14.191" "mkdir -p /workspace/magaphragma-connectome/experiments/phase6e" 2>&1 | Out-Null
  & scp.exe @scp "$Root\experiments\phase6e\AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json" "root@92.190.14.191:/workspace/magaphragma-connectome/experiments/phase6e/" 2>&1 | Out-Null
  # Only retarget if not already on 0/10 with util
  $fix = & ssh.exe @sshBase -p 28225 "root@92.190.14.191" "bash -s" @"
UTIL=`$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
PID=`$(pgrep -f run_affinity_fullvol_s7_fast_worker | head -1 || true)
SH=
[ -n "`$PID" ] && SH=`$(tr '\0' '\n' < /proc/`$PID/environ 2>/dev/null | grep '^S7_SHARD=' || true)
echo util=`$UTIL shard=`$SH
if [ "`${UTIL:-0}" -gt 5 ] && echo "`$SH" | grep -q '0/10'; then echo LEAVE_ALONE; exit 0; fi
echo NEED_RETARGET
pkill -f run_affinity_fullvol_s7_fast_worker 2>/dev/null || true
pkill -f run_vast_simple.sh 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-logs
cd /workspace/magaphragma-connectome
nohup env S7_SHARD=0/10 S7_WORKER_ID=vast-3060ti S7_TILE_BATCH=8 bash -c 'while true; do bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1; sleep 20; done' >/tmp/s7-logs/loop.log 2>&1 &
sleep 12
tail -20 /tmp/s7-logs/worker.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
"@ 2>&1 | Out-String
  [void]$out.AppendLine($fix)
  Set-Content $log $out.ToString()
} -ArgumentList $Key, $sshBase, $LogDir, $Root

Write-Host "Waiting jobs... logs=$LogDir"
$jobs | Wait-Job -Timeout 600 | Out-Null
$jobs | Receive-Job 2>$null | Out-Null
$jobs | Remove-Job -Force

Write-Host "==== SUMMARIES ===="
Get-ChildItem $LogDir -Filter *.log | ForEach-Object {
  Write-Host ("----- " + $_.Name + " -----")
  Get-Content $_.FullName -Tail 50
  Write-Host ""
}
Write-Host "DONE $LogDir"
