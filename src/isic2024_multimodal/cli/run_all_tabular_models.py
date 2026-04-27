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

from isic2024_multimodal.evaluation.metrics import PRIMARY_PAUC_METRIC
from isic2024_multimodal.training.reproducibility import DEFAULT_SEED
from isic2024_multimodal.utils.runtime_env import (
    DEFAULT_MLFLOW_FILE_TRACKING_URI,
    DEFAULT_MLFLOW_SQLITE_TRACKING_URI,
    ensure_expected_conda_env,
    load_project_env,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_MODELS = ["logistic_regression", "svm", "mlp", "xgboost", "catboost"]
DEFAULT_FEATURE_SETS = ["strict_base", "strict_fe", "strict_main_input"]


@dataclass
class RunningJob:
    device: int
    model_name: str
    process: subprocess.Popen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular baseline models in parallel across multiple GPUs.")
    parser.add_argument("--dataset-root", default="data/raw/isic_2024_challenge")
    parser.add_argument("--eda-dir", default="experiments/evidence/eda/isic_2024")
    parser.add_argument("--feature-set-json", default="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json")
    parser.add_argument("--output-root", default="experiments/outputs/tabular_baselines")
    parser.add_argument("--tracking-uri", default=get_repo_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Baselines")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--devices", nargs="*", type=int, default=None, help="Visible GPU indices to run in parallel.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--feature-sets", nargs="*", default=DEFAULT_FEATURE_SETS)
    parser.add_argument(
        "--leaderboard-output",
        default=None,
        help="Optional CSV leaderboard output path. Defaults to <output-root>/tabular_mlflow_leaderboard.csv.",
    )
    parser.add_argument(
        "--html-report-output",
        default=None,
        help="Optional HTML leaderboard output path. Defaults to <output-root>/tabular_mlflow_report.html.",
    )
    parser.add_argument("--skip-reports", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    ensure_expected_conda_env()

    if args.devices:
        validate_gpu_request(args.devices)
        failures = run_parallel(args)
    else:
        failures = run_sequential(args)

    report_failures = [] if args.skip_reports else generate_reports(args)
    failures.extend(report_failures)

    if failures:
        print("[run_all_tabular_models] Completed with failures:")
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
    for device in unique_devices:
        if device < 0 or device >= visible_count:
            raise RuntimeError(f"GPU id {device} is out of range for {visible_count} visible devices.")


def run_sequential(args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    for model_name in args.models:
        command = build_command(model_name, args, device=None)
        print(f"[run_all_tabular_models] Running {model_name}")
        result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            failures.append({"model": model_name, "returncode": result.returncode})
    return failures


def run_parallel(args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    pending = deque(args.models)
    active: dict[int, RunningJob] = {}
    devices = list(args.devices or [])

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            model_name = pending.popleft()
            command = build_command(model_name, args, device=device)
            env = build_subprocess_env(device=device)
            print(f"[run_all_tabular_models] Launching {model_name} on GPU {device}")
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
            active[device] = RunningJob(device=device, model_name=model_name, process=process)

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            print(f"[run_all_tabular_models] Finished {job.model_name} on GPU {device} (returncode={returncode})")
            if returncode != 0:
                failures.append({"model": job.model_name, "returncode": returncode})
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        if active:
            time.sleep(2)

    return failures


def build_command(model_name: str, args: argparse.Namespace, *, device: int | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_tabular_baseline",
        "--dataset-root",
        str(resolve_repo_path(args.dataset_root)),
        "--eda-dir",
        str(resolve_repo_path(args.eda_dir)),
        "--feature-set-json",
        str(resolve_repo_path(args.feature_set_json)),
        "--experiment-name",
        args.experiment_name,
        "--output-root",
        str(resolve_repo_path(args.output_root)),
        "--tracking-uri",
        args.tracking_uri,
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--validation-size",
        str(args.validation_size),
        "--models",
        model_name,
    ]
    if args.feature_sets:
        command.append("--feature-sets")
        command.extend(args.feature_sets)
    if device is not None:
        command.extend(["--device", "cuda"])
    return command


def generate_reports(args: argparse.Namespace) -> list[dict[str, str | int]]:
    output_root = resolve_repo_path(args.output_root)
    leaderboard_output = resolve_repo_path(args.leaderboard_output or output_root / "tabular_mlflow_leaderboard.csv")
    html_output = resolve_repo_path(args.html_report_output or output_root / "tabular_mlflow_report.html")

    report_specs = [
        {
            "name": "tabular_mlflow_leaderboard",
            "output": leaderboard_output,
            "command": [
                sys.executable,
                "-m",
                "isic2024_multimodal.reporting.mlflow_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--sort-metric",
                f"best_{PRIMARY_PAUC_METRIC}",
                "--output",
                str(leaderboard_output),
            ],
        },
        {
            "name": "tabular_mlflow_report_html",
            "output": html_output,
            "command": [
                sys.executable,
                "-m",
                "isic2024_multimodal.reporting.mlflow_html_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--parent-sort-metric",
                f"best_{PRIMARY_PAUC_METRIC}",
                "--child-sort-metric",
                f"val_{PRIMARY_PAUC_METRIC}",
                "--output",
                str(html_output),
            ],
        },
    ]

    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    for spec in report_specs:
        spec["output"].parent.mkdir(parents=True, exist_ok=True)
        print(f"[run_all_tabular_models] Generating {spec['name']} -> {spec['output']}")
        result = subprocess.run(spec["command"], check=False, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            failures.append({"model": spec["name"], "returncode": result.returncode})
    return failures


def build_subprocess_env(*, device: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(SRC_ROOT)
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    if current_pythonpath:
        entries = current_pythonpath.split(os.pathsep)
        if src_path not in entries:
            env["PYTHONPATH"] = os.pathsep.join([src_path, *entries])
    else:
        env["PYTHONPATH"] = src_path
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_repo_default_mlflow_tracking_uri() -> str:
    if Path(REPO_ROOT / "experiments/logs/mlflow.db").exists():
        return DEFAULT_MLFLOW_SQLITE_TRACKING_URI
    return DEFAULT_MLFLOW_FILE_TRACKING_URI


if __name__ == "__main__":
    main()
