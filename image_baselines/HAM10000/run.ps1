param(
  [string]$DatasetRoot = "",
  [string]$OutputRoot = "",
  [string]$MlflowTrackingUri = "file:./mlruns",
  [string]$ExperimentName = "ISIC2024-Image-Benchmark"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

function Resolve-PythonCommand {
  $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }

  return "python"
}

$PythonCommand = Resolve-PythonCommand

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
  $DatasetRoot = Join-Path $RepoRoot "dataset"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepoRoot "artifacts"
}

& $PythonCommand -m isic2024_benchmark.run_experiment `
  --config (Join-Path $ScriptDir "config.json") `
  --dataset-root $DatasetRoot `
  --output-root $OutputRoot `
  --mlflow-tracking-uri $MlflowTrackingUri `
  --experiment-name $ExperimentName
