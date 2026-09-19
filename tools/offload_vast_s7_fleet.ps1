# Offload completed Affinity chunks from multiple Vast hosts.
param(
  [switch]$Loop,
  [int]$IntervalSec = 120,
  [string]$HostsFile = ""
)
$ErrorActionPreference = "Stop"
$Key = if ($env:VAST_KEY) { $env:VAST_KEY } else { "$env:USERPROFILE\.ssh\vast_affinity" }
$LocalChunks = if ($env:LOCAL_CHUNKS) { $env:LOCAL_CHUNKS } else { "$env:USERPROFILE\s7-affinity-out\chunks" }
$Log = if ($env:LOG) { $env:LOG } else { "$env:USERPROFILE\s7-affinity-out\vast_offload_fleet_win.log" }
$KeepRemote = if ($env:KEEP_REMOTE) { $env:KEEP_REMOTE } else { "0" }
$LowDiskKb = 2500000
$SingleScript = Join-Path $PSScriptRoot "offload_vast_s7_chunks.ps1"

# Default fleet: existing 3060 Ti + known new hosts from latest launch (override via VAST_FLEET or HostsFile).
$fleet = @()
if ($HostsFile -and (Test-Path $HostsFile)) {
  $fleet = Get-Content $HostsFile | Where-Object { $_ -and $_ -notmatch '^\s*#' } | ForEach-Object {
    $p = $_.Trim() -split '[:\s]+'
    [pscustomobject]@{ Host = $p[0]; Port = $p[1] }
  }
} elseif ($env:VAST_FLEET) {
  $fleet = $env:VAST_FLEET -split ';' | Where-Object { $_ } | ForEach-Object {
    $p = $_.Trim() -split ':'
    [pscustomobject]@{ Host = $p[0]; Port = $p[1] }
  }
} else {
  $fleet = @(
    [pscustomobject]@{ Host = "92.190.14.191"; Port = "28225" }
    [pscustomobject]@{ Host = "14.227.95.149"; Port = "35774" }
    [pscustomobject]@{ Host = "122.51.254.66"; Port = "22704" }
    [pscustomobject]@{ Host = "74.96.198.230"; Port = "22706" }
    [pscustomobject]@{ Host = "202.122.49.242"; Port = "22708" }
    [pscustomobject]@{ Host = "38.29.145.27"; Port = "22712" }
  )
}

New-Item -ItemType Directory -Force -Path $LocalChunks | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")), $msg
  Add-Content -Path $Log -Value $line
  Write-Host $line
}

function Invoke-OneHost([string]$h, [string]$p) {
  Write-Log "offload host=$h port=$p"
  $env:VAST_HOST = $h
  $env:VAST_PORT = $p
  $env:VAST_KEY = $Key
  $env:LOCAL_CHUNKS = $LocalChunks
  $env:KEEP_REMOTE = $KeepRemote
  & $SingleScript
}

do {
  foreach ($f in $fleet) {
    try { Invoke-OneHost $f.Host $f.Port }
    catch { Write-Log "ERROR $($f.Host):$($f.Port) $_" }
  }
  if (-not $Loop) { break }
  Start-Sleep -Seconds $IntervalSec
} while ($true)
