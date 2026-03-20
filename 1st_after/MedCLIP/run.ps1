$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
python -m cbis_ddsm_benchmark.run_experiment `
  --config (Join-Path $ScriptDir "config.json") `
  --dataset-root (Join-Path $RepoRoot "dataset\archive_CBIS-DDSM_kaggle") `
  --output-root (Join-Path $RepoRoot "artifacts") `
  --mlflow-tracking-uri "file:./mlruns" `
  --experiment-name "CBIS-DDSM-Benchmark"
