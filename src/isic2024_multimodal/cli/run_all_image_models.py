#!/usr/bin/env python
from __future__ import annotations

import argparse
import atexit
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


@dataclass
class RunningJob:
    device: int
    config_path: Path
    process: subprocess.Popen
    started_at: float


def make_run_group_id(prefix: str = "image_all") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every image baseline config.")
    parser.add_argument("--config-root", default="experiments/configs/image_baselines")
    parser.add_argument("--dataset-root", default="data/raw/isic_2024_challenge")
    parser.add_argument("--output-root", default="experiments/outputs/image_baselines")
    parser.add_argument(
        "--tracking-uri",
        default=get_repo_default_mlflow_tracking_uri(),
    )
    parser.add_argument("--experiment-name", default="ISIC2024-Image-Baselines")
    parser.add_argument("--run-group-id", default=None, help="Optional MLflow run group tag. Defaults to a timestamp.")
    parser.add_argument("--dataset-id", default=None, help="Versioned dataset id for registry/report filtering.")
    parser.add_argument("--dataset-spec", default=None, help="Dataset spec JSON path used for this run.")
    parser.add_argument("--model-family", default="image_baselines", help="Experiment family tag.")
    parser.add_argument("--holdout-split-csv", default="data/splits/isic2024_train_validation_test_split_seed42.csv")
    parser.add_argument("--cv-split-csv", default="data/splits/isic2024_train_validation_5fold_seed42.csv")
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--devices", nargs="*", type=int, default=None, help="Visible GPU indices to run in parallel.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model directory names to include.")
    parser.add_argument("--exclude-models", nargs="*", default=None, help="Optional model directory names to exclude.")
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
    command_start = time.time()
    load_project_env()
    args = parse_args()
    args.run_group_id = args.run_group_id or make_run_group_id()
    command_status = {"value": "failed"}

    def log_command_end() -> None:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [run_all_image_models] "
            f"End status={command_status['value']} run_group_id={args.run_group_id} "
            f"duration={time.time() - command_start:.1f}s",
            flush=True,
        )

    atexit.register(log_command_end)
    ensure_expected_conda_env()
    base_dir = resolve_repo_path(args.config_root)
    config_paths = sorted(base_dir.glob("*/config.json"))
    config_paths = filter_config_paths(
        config_paths,
        models=args.models,
        exclude_models=args.exclude_models,
    )
    if not config_paths:
        raise RuntimeError(f"No model config files found under {base_dir}.")

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
    command_status["value"] = "ok"


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
    env = build_subprocess_env()
    for config_path in config_paths:
        command = build_command(config_path, args, device=None)
        print(f"[run_all_models] Running {config_path.parent.name}")
        result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
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
            env = build_subprocess_env(device=device)
            started_at = time.time()
            print(f"[run_all_models] Launching {config_path.parent.name} on GPU {device}")
            process = subprocess.Popen(command, env=env, cwd=REPO_ROOT)
            active[device] = RunningJob(device=device, config_path=config_path, process=process, started_at=started_at)

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            print(
                f"[run_all_models] Finished {job.config_path.parent.name} on GPU {device} "
                f"(returncode={returncode}, duration={time.time() - job.started_at:.1f}s)"
            )
            if returncode != 0:
                failures.append({"model": job.config_path.parent.name, "returncode": returncode})
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        if active:
            time.sleep(2)

    return failures


def filter_config_paths(
    config_paths: list[Path],
    *,
    models: list[str] | None,
    exclude_models: list[str] | None,
) -> list[Path]:
    selected = list(config_paths)
    if models:
        wanted = set(models)
        selected = [path for path in selected if path.parent.name in wanted]
    if exclude_models:
        blocked = set(exclude_models)
        selected = [path for path in selected if path.parent.name not in blocked]
    return selected


def build_command(config_path: Path, args: argparse.Namespace, *, device: int | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_image_baseline",
        "--config",
        str(config_path),
        "--dataset-root",
        str(resolve_repo_path(args.dataset_root)),
        "--output-root",
        str(resolve_repo_path(args.output_root)),
        "--mlflow-tracking-uri",
        args.tracking_uri,
        "--experiment-name",
        args.experiment_name,
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        args.dataset_id or "",
        "--dataset-spec",
        args.dataset_spec or "",
        "--model-family",
        args.model_family,
        "--holdout-split-csv",
        str(resolve_repo_path(args.holdout_split_csv)),
        "--cv-split-csv",
        str(resolve_repo_path(args.cv_split_csv)),
        "--cv-fold",
        str(args.cv_fold),
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
    output_root = resolve_repo_path(args.output_root)
    leaderboard_output = resolve_repo_path(args.leaderboard_output or output_root / "image_mlflow_leaderboard.csv")
    html_output = resolve_repo_path(args.html_report_output or output_root / "image_mlflow_report.html")

    report_specs = [
        {
            "name": "image_mlflow_leaderboard",
            "output": leaderboard_output,
            "command": [
                sys.executable,
                "-m",
                "isic2024_multimodal.reporting.mlflow_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--run-group-id",
                args.run_group_id,
                "--dataset-id",
                args.dataset_id or "",
                "--model-family",
                args.model_family,
                "--sort-metric",
                f"best_{PRIMARY_PAUC_METRIC}",
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
                "isic2024_multimodal.reporting.mlflow_html_report",
                "--tracking-uri",
                args.tracking_uri,
                "--experiment-name",
                args.experiment_name,
                "--run-group-id",
                args.run_group_id,
                "--dataset-id",
                args.dataset_id or "",
                "--model-family",
                args.model_family,
                "--parent-sort-metric",
                f"best_{PRIMARY_PAUC_METRIC}",
                "--child-sort-metric",
                f"best_val_{PRIMARY_PAUC_METRIC}",
                "--output",
                str(html_output),
            ],
        },
    ]

    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    for spec in report_specs:
        spec["output"].parent.mkdir(parents=True, exist_ok=True)
        print(f"[run_all_models] Generating {spec['name']} -> {spec['output']}")
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
    explicit_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit_uri:
        return explicit_uri
    if (REPO_ROOT / "experiments/logs/mlflow.db").exists():
        return DEFAULT_MLFLOW_SQLITE_TRACKING_URI
    return DEFAULT_MLFLOW_FILE_TRACKING_URI


if __name__ == "__main__":
    main()
