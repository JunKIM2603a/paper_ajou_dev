from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRANSFER_STRATEGIES = ("full_finetune", "linear_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every model config under 1st_after.")
    parser.add_argument("--dataset-root", default="dataset/archive_CBIS-DDSM_kaggle")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="CBIS-DDSM-Benchmark")
    parser.add_argument("--transfer-strategy", choices=TRANSFER_STRATEGIES, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path("1st_after")
    config_paths = sorted(base_dir.glob("*/config.json"))
    if not config_paths:
        raise RuntimeError("No model config files found under 1st_after.")

    for config_path in config_paths:
        command = [
            sys.executable,
            "-m",
            "cbis_ddsm_benchmark.run_experiment",
            "--config",
            str(config_path),
            "--dataset-root",
            args.dataset_root,
            "--output-root",
            args.output_root,
            "--mlflow-tracking-uri",
            args.tracking_uri,
            "--experiment-name",
            args.experiment_name,
        ]
        if args.transfer_strategy is not None:
            command.extend(["--transfer-strategy", args.transfer_strategy])
        print(f"[run_all_models] Running {config_path.parent.name}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
