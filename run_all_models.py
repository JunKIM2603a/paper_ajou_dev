#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from isic2024_benchmark.reproducibility import DEFAULT_SEED
from isic2024_benchmark.runtime_env import ensure_expected_conda_env


@dataclass
class RunningJob:
    device: int
    config_path: Path
    process: subprocess.Popen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every model config under image_baselines.")
    parser.add_argument("--dataset-root", default="dataset/isic-2024-challenge")
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
    parser.add_argument("--devices", nargs="*", type=int, default=None, help="Visible GPU indices to run in parallel.")
    parser.add_argument(
        "--leaderboard-output",
        default=None,
        help="Optional CSV leaderboard output path. Defaults to <output-root>/image_mlflow_leaderboard.csv.",
    )
    parser.add_argument(
        "--html-report-output",
        default=None,
        help="Optional HTML leaderboard output path. Defaults to <output-root>/image_mlflow_report.html.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip post-run MLflow CSV/HTML report generation.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    args = parse_args()
    base_dir = Path("image_baselines")
    config_paths = sorted(base_dir.glob("*/config.json"))
    if not config_paths:
        raise RuntimeError("No model config files found under image_baselines.")

    if args.devices:
        validate_gpu_request(args.devices)
        failures = run_parallel(config_paths, args)
    else:
        failures = run_sequential(config_paths, args)

    report_failures = [] if args.skip_reports else generate_reports(args)
    failures.extend(report_failures)

    if failures:
        print("[run_all_models] Completed with failures:")
        for failure in failures:
            print(f"  - {failure['model']} (returncode={failure['returncode']})")
        raise SystemExit(1)


def validate_gpu_request(devices: list[int]) -> None:
    import torch

    unique_devices = sorted(set(devices))
    if len(unique_devices) != len(devices):
        raise RuntimeError(f"Duplicate GPU ids are not allowed: {devices}")
    if not torch.cuda.is_available():
        raise RuntimeError("`--devices` was provided, but torch.cuda.is_available() is False.")
    visible_count = torch.cuda.device_count()
    if visible_count < len(unique_devices):
        raise RuntimeError(
            f"Requested {len(unique_devices)} GPUs via --devices {unique_devices}, "
            f"but only {visible_count} visible CUDA devices are available."
        )
    for device in unique_devices:
        if device < 0:
            raise RuntimeError(f"GPU ids must be non-negative, got {device}")
        if device >= visible_count:
            raise RuntimeError(f"GPU id {device} is out of range for {visible_count} visible devices.")


def run_sequential(config_paths: list[Path], args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    for config_path in config_paths:
        command = build_command(config_path, args, device=None)
        print(f"[run_all_models] Running {config_path.parent.name}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append({"model": config_path.parent.name, "returncode": result.returncode})
            print(f"[run_all_models] Model failed: {config_path.parent.name} (returncode={result.returncode})")
    return failures


def run_parallel(config_paths: list[Path], args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    pending = deque(config_paths)
    active: dict[int, RunningJob] = {}
    devices = list(args.devices or [])

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            config_path = pending.popleft()
            command = build_command(config_path, args, device=device)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            print(f"[run_all_models] Launching {config_path.parent.name} on GPU {device}")
            process = subprocess.Popen(command, env=env)
            active[device] = RunningJob(device=device, config_path=config_path, process=process)

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            print(f"[run_all_models] Finished {job.config_path.parent.name} on GPU {device} (returncode={returncode})")
            if returncode != 0:
                failures.append({"model": job.config_path.parent.name, "returncode": returncode})
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        if active:
            time.sleep(2)

    return failures


def build_command(config_path: Path, args: argparse.Namespace, *, device: int | None) -> list[str]:
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
    if device is not None:
        command.extend(["--device", "cuda"])
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
    return command


def generate_reports(args: argparse.Namespace) -> list[dict[str, str | int]]:
    leaderboard_output = Path(args.leaderboard_output or Path(args.output_root) / "image_mlflow_leaderboard.csv")
    html_output = Path(args.html_report_output or Path(args.output_root) / "image_mlflow_report.html")

    report_specs = [
        {
            "name": "image_mlflow_leaderboard",
            "output": leaderboard_output,
            "command": [
                sys.executable,
                "-m",
                "isic2024_benchmark.mlflow_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--sort-metric",
                "best_auc_roc",
                "--output",
                str(leaderboard_output),
            ],
        },
        {
            "name": "image_mlflow_report_html",
            "output": html_output,
            "command": [
                sys.executable,
                "-m",
                "isic2024_benchmark.mlflow_html_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--parent-sort-metric",
                "best_auc_roc",
                "--child-sort-metric",
                "test_auc_roc",
                "--output",
                str(html_output),
            ],
        },
    ]

    failures: list[dict[str, str | int]] = []
    for spec in report_specs:
        spec["output"].parent.mkdir(parents=True, exist_ok=True)
        print(f"[run_all_models] Generating {spec['name']} -> {spec['output']}")
        result = subprocess.run(spec["command"], check=False)
        if result.returncode != 0:
            failures.append({"model": spec["name"], "returncode": result.returncode})
    return failures


if __name__ == "__main__":
    main()
