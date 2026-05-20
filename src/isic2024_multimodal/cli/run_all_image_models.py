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
from isic2024_multimodal.utils.device import resolve_device_list
from isic2024_multimodal.utils.progress import (
    estimate_remaining_seconds,
    format_eta,
    format_progress_duration,
    progress_index_label,
)
from isic2024_multimodal.utils.runtime_env import (
    DEFAULT_MLFLOW_FILE_TRACKING_URI,
    DEFAULT_MLFLOW_SQLITE_TRACKING_URI,
    ensure_expected_conda_env,
    load_project_env,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
PROGRESS_HEARTBEAT_SECONDS = 60.0


@dataclass
class RunningJob:
    device: int
    plan_item: "ImageJobPlanItem"
    process: subprocess.Popen
    started_at: float


@dataclass(frozen=True)
class ImageJobPlanItem:
    job_index: int
    total_jobs: int
    model_index: int
    total_models: int
    config_path: Path

    @property
    def model_name(self) -> str:
        return self.config_path.parent.name


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
    parser.add_argument("--split-protocol", choices=["nested_cv", "legacy_holdout"], default="nested_cv")
    parser.add_argument("--nested-split-csv", default="data/splits/isic2024_official_train_nested_5x4_seed42.csv")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
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
    parser.add_argument("--auto-download-checkpoints", action="store_true")
    parser.add_argument("--devices", nargs="*", type=int, default=None, help="Visible GPU indices to run in parallel.")
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu"],
        default="auto",
        help="Device policy for subprocesses. auto prefers CUDA and falls back to CPU; cpu forces CPU.",
    )
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
        log_event(
            f"End status={command_status['value']} run_group_id={args.run_group_id} "
            f"duration={format_progress_duration(time.time() - command_start)}"
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

    log_event(f"requested_devices={args.devices or 'auto'} device_policy={args.device_policy}")
    device_resolution = resolve_device_list(args.devices, device_policy=args.device_policy)
    args.resolved_devices = device_resolution.resolved_devices
    args.device_fallback_reason = device_resolution.fallback_reason
    log_event(
        f"resolved_devices={args.resolved_devices or 'cpu'} "
        f"cuda_available={device_resolution.cuda_available} "
        f"visible_device_count={device_resolution.visible_device_count} "
        f"fallback={args.device_fallback_reason or 'none'}"
    )

    if args.resolved_devices:
        failures = run_parallel(config_paths, args)
    else:
        failures = run_sequential(config_paths, args)

    report_failures = [] if args.skip_reports else generate_reports(args)
    failures.extend(report_failures)

    if failures:
        log_event("Completed with failures:")
        for failure in failures:
            print(f"  - {failure['model']} (returncode={failure['returncode']})")
        raise SystemExit(1)
    command_status["value"] = "ok"


def run_sequential(config_paths: list[Path], args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    plan = build_image_job_plan(config_paths)
    completed_jobs = 0
    suite_start = time.time()
    for item in plan:
        command = build_command(item.config_path, args, device=None)
        job_start = time.time()
        log_event(
            "Start "
            f"{format_image_job_progress(item)} device=cpu "
            f"completed_jobs={completed_jobs} pending_jobs={item.total_jobs - item.job_index + 1} "
            f"elapsed={format_progress_duration(time.time() - suite_start)} "
            f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=completed_jobs, total_count=item.total_jobs)}"
        )
        result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
        completed_jobs += 1
        log_event(
            "Finished "
            f"{format_image_job_progress(item)} device=cpu returncode={result.returncode} "
            f"duration={format_progress_duration(time.time() - job_start)} "
            f"completed_jobs={completed_jobs}/{item.total_jobs} "
            f"elapsed={format_progress_duration(time.time() - suite_start)} "
            f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=completed_jobs, total_count=item.total_jobs)}"
        )
        if result.returncode != 0:
            failures.append({"model": item.model_name, "returncode": result.returncode})
            log_event(
                f"Model failed model_index={progress_index_label(item.model_index, item.total_models)} "
                f"model={item.model_name} returncode={result.returncode}"
            )
    return failures


def run_parallel(config_paths: list[Path], args: argparse.Namespace) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    plan = build_image_job_plan(config_paths)
    pending = deque(plan)
    active: dict[int, RunningJob] = {}
    devices = list(args.resolved_devices or [])
    completed_job_indices: set[int] = set()
    suite_start = time.time()
    last_heartbeat = 0.0

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            item = pending.popleft()
            command = build_command(item.config_path, args, device=device)
            env = build_subprocess_env(device=device)
            started_at = time.time()
            log_event(
                "Start "
                f"{format_image_job_progress(item)} gpu={device} "
                f"active_jobs={len(active) + 1} pending_jobs={len(pending)} "
                f"elapsed={format_progress_duration(time.time() - suite_start)} "
                f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=len(completed_job_indices), total_count=item.total_jobs)}"
            )
            process = subprocess.Popen(command, env=env, cwd=REPO_ROOT)
            active[device] = RunningJob(device=device, plan_item=item, process=process, started_at=started_at)

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            item = job.plan_item
            completed_job_indices.add(item.job_index)
            log_event(
                "Finished "
                f"{format_image_job_progress(item)} gpu={device} returncode={returncode} "
                f"duration={format_progress_duration(time.time() - job.started_at)} "
                f"completed_jobs={len(completed_job_indices)}/{item.total_jobs} "
                f"elapsed={format_progress_duration(time.time() - suite_start)} "
                f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=len(completed_job_indices), total_count=item.total_jobs)}"
            )
            if returncode != 0:
                failures.append({"model": item.model_name, "returncode": returncode})
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        now = time.time()
        if active and now - last_heartbeat >= PROGRESS_HEARTBEAT_SECONDS:
            log_image_parallel_heartbeat(
                plan=plan,
                active=active,
                completed_job_indices=completed_job_indices,
                pending_count=len(pending),
                suite_start=suite_start,
            )
            last_heartbeat = now

        if active:
            time.sleep(2)

    return failures


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_event(message: str) -> None:
    print(f"[{current_timestamp()}] [run_all_image_models] {message}", flush=True)


def build_image_job_plan(config_paths: list[Path]) -> list[ImageJobPlanItem]:
    total_jobs = len(config_paths)
    return [
        ImageJobPlanItem(
            job_index=index,
            total_jobs=total_jobs,
            model_index=index,
            total_models=total_jobs,
            config_path=config_path,
        )
        for index, config_path in enumerate(config_paths, start=1)
    ]


def format_image_job_progress(item: ImageJobPlanItem) -> str:
    return (
        f"Progress job={progress_index_label(item.job_index, item.total_jobs)} "
        f"model={progress_index_label(item.model_index, item.total_models)} {item.model_name}"
    )


def log_image_parallel_heartbeat(
    *,
    plan: list[ImageJobPlanItem],
    active: dict[int, RunningJob],
    completed_job_indices: set[int],
    pending_count: int,
    suite_start: float,
) -> None:
    elapsed = time.time() - suite_start
    completed_models = [
        item.model_name for item in plan if item.job_index in completed_job_indices
    ]
    active_models = [job.plan_item.model_name for job in active.values()]
    pending_models = [
        item.model_name
        for item in plan
        if item.job_index not in completed_job_indices and item.model_name not in active_models
    ]
    log_event(
        "Heartbeat "
        f"completed_jobs={len(completed_job_indices)}/{len(plan)} active_jobs={len(active)} pending_jobs={pending_count} "
        f"completed_models={','.join(completed_models) or 'none'} "
        f"active_models={','.join(active_models) or 'none'} "
        f"pending_models={','.join(pending_models) or 'none'} "
        f"elapsed={format_progress_duration(elapsed)} "
        f"eta={format_progress_duration(estimate_remaining_seconds(elapsed, len(completed_job_indices), len(plan)))}"
    )
    for device, job in sorted(active.items()):
        item = job.plan_item
        log_event(
            f"GPU {device} active model={item.model_name} "
            f"model_index={progress_index_label(item.model_index, item.total_models)} "
            f"job={progress_index_label(item.job_index, item.total_jobs)} "
            f"elapsed={format_progress_duration(time.time() - job.started_at)}"
        )


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
    split_protocol = getattr(args, "split_protocol", "nested_cv")
    nested_split_csv = getattr(args, "nested_split_csv", "data/splits/isic2024_official_train_nested_5x4_seed42.csv")
    outer_fold = getattr(args, "outer_fold", 0)
    inner_fold = getattr(args, "inner_fold", 0)
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
        "--split-protocol",
        split_protocol,
        "--nested-split-csv",
        str(resolve_repo_path(nested_split_csv)),
        "--outer-fold",
        str(outer_fold),
        "--inner-fold",
        str(inner_fold),
        "--holdout-split-csv",
        str(resolve_repo_path(args.holdout_split_csv)),
        "--cv-split-csv",
        str(resolve_repo_path(args.cv_split_csv)),
        "--cv-fold",
        str(args.cv_fold),
        "--seed",
        str(args.seed),
    ]
    device_arg = command_device_arg(args, device)
    if device_arg is not None:
        command.extend(["--device", device_arg])
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
    if args.auto_download_checkpoints:
        command.append("--auto-download-checkpoints")
    return command


def command_device_arg(args: argparse.Namespace, device: int | None) -> str | None:
    if device is not None:
        return "cuda"
    if getattr(args, "device_policy", "auto") == "cpu":
        return "cpu"
    if getattr(args, "devices", None) and not getattr(args, "resolved_devices", []):
        return "cpu"
    return None


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
        report_start = time.time()
        log_event(f"Start report={spec['name']} output={spec['output']}")
        result = subprocess.run(spec["command"], check=False, cwd=REPO_ROOT, env=env)
        log_event(
            f"Finished report={spec['name']} output={spec['output']} returncode={result.returncode} "
            f"duration={format_progress_duration(time.time() - report_start)}"
        )
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
