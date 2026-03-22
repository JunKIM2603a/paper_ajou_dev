param(
  [string]$TrackingDir = ".\mlruns",
  [string]$DatabasePath = ".\mlflow.db",
  [string]$OutputDir = ".\artifacts",
  [switch]$ClearCache,
  [switch]$StopMlflow
)

$ErrorActionPreference = "Stop"

function Stop-MLflowProcesses {
  $targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -like "*-m mlflow ui*" -or
      $_.CommandLine -like "*-m mlflow server*" -or
      $_.CommandLine -like "*mlflow.server.fastapi_app:app*" -or
      $_.CommandLine -like "*start_mlflow.ps1*"
    )
  }

  foreach ($process in $targets) {
    Write-Host "[reset_results] stopping MLflow process $($process.ProcessId)"
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  }

  if ($targets) {
    Start-Sleep -Seconds 2
  }
}

function Remove-FileWithRetry {
  param(
    [string]$Path,
    [int]$Attempts = 5,
    [int]$DelaySeconds = 2
  )

  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Force
      return
    }
    catch {
      if ($attempt -eq $Attempts) {
        throw
      }
      Write-Host "[reset_results] waiting for file release: $Path (attempt $attempt/$Attempts)"
      Start-Sleep -Seconds $DelaySeconds
    }
  }
}

Write-Host "[reset_results] tracking dir: $TrackingDir"
Write-Host "[reset_results] database path: $DatabasePath"
Write-Host "[reset_results] output dir: $OutputDir"
Write-Host "[reset_results] clear cache: $ClearCache"
Write-Host "[reset_results] stop mlflow: $StopMlflow"

if ($StopMlflow) {
  Stop-MLflowProcesses
}

if (Test-Path -LiteralPath $TrackingDir) {
  Remove-Item -LiteralPath $TrackingDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TrackingDir -Force | Out-Null

if (Test-Path -LiteralPath $DatabasePath) {
  try {
    Remove-FileWithRetry -Path $DatabasePath
  }
  catch {
    throw "Failed to remove '$DatabasePath'. Stop the running MLflow server first, or rerun with -StopMlflow."
  }
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
  New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
elseif (-not $ClearCache) {
  Get-ChildItem -LiteralPath $OutputDir -Force | Where-Object { $_.Name -ne "cache" } | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
  }
}
else {
  Remove-Item -LiteralPath $OutputDir -Recurse -Force
  New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

Write-Host "[reset_results] done"
