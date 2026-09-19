# Windows PowerShell Vast to local offload (no WSL Ubuntu required)
param(
  [switch]$Loop,
  [int]$IntervalSec = 120
)
$ErrorActionPreference = "Stop"
$HostName = if ($env:VAST_HOST) { $env:VAST_HOST } else { "92.190.14.191" }
$Port = if ($env:VAST_PORT) { $env:VAST_PORT } else { "28225" }
$Key = if ($env:VAST_KEY) { $env:VAST_KEY } else { "$env:USERPROFILE\.ssh\vast_affinity" }
$LocalChunks = if ($env:LOCAL_CHUNKS) { $env:LOCAL_CHUNKS } else { "$env:USERPROFILE\s7-affinity-out\chunks" }
$Log = if ($env:LOG) { $env:LOG } else { "$env:USERPROFILE\s7-affinity-out\vast_offload_win.log" }
$KeepRemote = if ($env:KEEP_REMOTE) { $env:KEEP_REMOTE } else { "0" }
$LowDiskKb = 2500000

New-Item -ItemType Directory -Force -Path $LocalChunks | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")), $msg
  Add-Content -Path $Log -Value $line
  Write-Host $line
}

function Invoke-Vast([string]$remoteCmd) {
  & ssh -p $Port -i $Key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=NUL -o LogLevel=ERROR -o ConnectTimeout=20 "root@$HostName" $remoteCmd
}

function Get-FreeKb {
  [int64]((Invoke-Vast "df -Pk / | awk 'NR==2{print `$4}'") -replace '\r','').Trim()
}

function Get-Completed {
  # Single-line remote shell — no temp scripts, no CRLF issues
  $remote = 'for d in /workspace/s7-out/chunks/MV-CHUNK-*; do [ -d "$d" ] || continue; [ -f "$d/receipt.json" ] || continue; [ -f "$d/affinities_core_czyx.npy" ] || continue; [ -f "$d/boundaries_core.tif" ] || continue; asz=$(stat -c%s "$d/affinities_core_czyx.npy" 2>/dev/null || echo 0); [ "$asz" -gt 100000000 ] || continue; basename "$d"; done'
  Invoke-Vast $remote
}

function Invoke-OffloadOnce {
  $avail = Get-FreeKb
  Write-Log "free_kb=$avail keep_remote=$KeepRemote"
  $done = @(Get-Completed | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
  Write-Log "completed_on_vast=$($done.Count)"
  foreach ($cid in $done) {
    $dest = Join-Path $LocalChunks $cid
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Log "scp $cid -> $dest"
    $remote = "root@${HostName}:/workspace/s7-out/chunks/$cid/"
    & scp -P $Port -i $Key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=NUL -o LogLevel=ERROR -r `
      "${remote}affinities_core_czyx.npy" "${remote}boundaries_core.tif" "${remote}receipt.json" $dest
    if ($LASTEXITCODE -ne 0) { Write-Log "ERROR scp failed $cid"; continue }
    $localSz = (Get-Item (Join-Path $dest "affinities_core_czyx.npy")).Length
    $remoteSz = [int64]((Invoke-Vast "stat -c%s /workspace/s7-out/chunks/$cid/affinities_core_czyx.npy") -replace '\r','').Trim()
    if ($localSz -ne $remoteSz) { Write-Log "ERROR size mismatch $cid local=$localSz remote=$remoteSz"; continue }
    if ($KeepRemote -eq "1") { Write-Log "verified $cid KEEP_REMOTE"; continue }
    Write-Log "verified $cid - deleting remote"
    Invoke-Vast "rm -rf /workspace/s7-out/chunks/$cid" | Out-Null
  }
  $avail = Get-FreeKb
  Write-Log "done free_kb=$avail"
  if ($avail -lt $LowDiskKb) { Write-Log "WARN still below LOW_DISK" }
}

do {
  try { Invoke-OffloadOnce } catch { Write-Log ("WARN " + $_.Exception.Message) }
  if (-not $Loop) { break }
  Start-Sleep -Seconds $IntervalSec
} while ($true)
