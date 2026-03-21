param(
  [string]$DatasetRoot = "",
  [string]$OutputRoot = "",
  [string]$MlflowTrackingUri = "file:./mlruns",
  [string]$ExperimentName = "CBIS-DDSM-Benchmark"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
  $DatasetRoot = Join-Path $RepoRoot "dataset"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepoRoot "artifacts"
}

python -m cbis_ddsm_benchmark.run_experiment `
  --config (Join-Path $ScriptDir "config.json") `
  --dataset-root $DatasetRoot `
  --output-root $OutputRoot `
  --mlflow-tracking-uri $MlflowTrackingUri `
  --experiment-name $ExperimentName
