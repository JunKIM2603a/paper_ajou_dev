from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from isic2024_benchmark.reproducibility import DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every model config under image_baselines.")
    parser.add_argument("--dataset-root", default="dataset/ISIC2024")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )
    parser.add_argument("--experiment-name", default="ISIC2024-Image-Benchmark")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--disable-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path("image_baselines")
    config_paths = sorted(base_dir.glob("*/config.json"))
    if not config_paths:
        raise RuntimeError("No model config files found under image_baselines.")

    failures: list[dict[str, str | int]] = []
    for config_path in config_paths:
        command = [
            sys.executable,
            "-m",
            "isic2024_benchmark.run_experiment",
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
            "--seed",
            str(args.seed),
        ]
        if args.max_trials is not None:
            command.extend(["--max-trials", str(args.max_trials)])
        if args.epochs_override is not None:
            command.extend(["--epochs-override", str(args.epochs_override)])
        if args.max_train_samples is not None:
            command.extend(["--max-train-samples", str(args.max_train_samples)])
        if args.max_val_samples is not None:
            command.extend(["--max-val-samples", str(args.max_val_samples)])
        if args.max_test_samples is not None:
            command.extend(["--max-test-samples", str(args.max_test_samples)])
        if args.disable_pretrained:
            command.append("--disable-pretrained")
        print(f"[run_all_models] Running {config_path.parent.name}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append({"model": config_path.parent.name, "returncode": result.returncode})
            print(
                f"[run_all_models] Model failed: {config_path.parent.name} "
                f"(returncode={result.returncode})"
            )

    if failures:
        print("[run_all_models] Completed with failures:")
        for failure in failures:
            print(
                f"  - {failure['model']} (returncode={failure['returncode']})"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
