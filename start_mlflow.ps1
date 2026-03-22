param(
  [string]$BackendStoreUri = ".\mlruns",
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
  $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }

  return "python"
}

$pythonCommand = Resolve-PythonCommand
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
  $PSScriptRoot
} else {
  "$PSScriptRoot;$env:PYTHONPATH"
}

Write-Host "[start_mlflow] backend store: $BackendStoreUri"
Write-Host "[start_mlflow] python: $pythonCommand"
Write-Host "[start_mlflow] pythonpath: $env:PYTHONPATH"
Write-Host "[start_mlflow] listening on http://${BindHost}:$Port"

& $pythonCommand -m mlflow ui `
  --backend-store-uri $BackendStoreUri `
  --host $BindHost `
  --port $Port
