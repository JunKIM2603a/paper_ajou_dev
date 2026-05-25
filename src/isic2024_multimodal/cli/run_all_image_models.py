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
    plan_item: "JobPlanItem"
    process: subprocess.Popen
    started_at: float


@dataclass(frozen=True)
class FoldSelection:
    outer_fold: int | None = None
    inner_fold: int | None = None
    cv_fold: int | None = None

    @property
    def label(self) -> str:
        if self.outer_fold is not None and self.inner_fold is not None:
            return f"outer_{self.outer_fold:02d}_inner_{self.inner_fold:02d}"
        if self.cv_fold is not None:
            return f"cv_{self.cv_fold:02d}"
        return "single_fold"


@dataclass(frozen=True)
class JobPlanItem:
    job_index: int
    total_jobs: int
    model_index: int
    total_models: int
    fold_index: int
    total_folds: int
    config_path: Path
    fold: FoldSelection


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
    parser.add_argument(
        "--all-folds",
        "--all-nested-folds",
        action="store_true",
        dest="all_folds",
        help=(
            "Run every fold found in the split artifact. For nested_cv this runs every "
            "(outer_fold, inner_fold) pair; for legacy_holdout this runs every cv_fold. "
            "Fold outputs are written under <output-root>/<fold_label>/ to avoid overwrites."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--disable-pretrained", action="store_true")
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

    print(
        f"[run_all_models] requested_devices={args.devices or 'auto'} device_policy={args.device_policy}",
        flush=True,
    )
    device_resolution = resolve_device_list(args.devices, device_policy=args.device_policy)
    args.resolved_devices = device_resolution.resolved_devices
    args.device_fallback_reason = device_resolution.fallback_reason
    print(
        "[run_all_models] "
        f"resolved_devices={args.resolved_devices or 'cpu'} "
        f"cuda_available={device_resolution.cuda_available} "
        f"visible_device_count={device_resolution.visible_device_count} "
        f"fallback={args.device_fallback_reason or 'none'}",
        flush=True,
    )
    fold_selections = resolve_fold_selections(args)
    print(
        "[run_all_models] "
        f"resolved_folds={','.join(fold.label for fold in fold_selections)} "
        f"count={len(fold_selections)}",
        flush=True,
    )

    if args.resolved_devices:
        failures = run_parallel(config_paths, args, fold_selections=fold_selections)
    else:
        failures = run_sequential(config_paths, args, fold_selections=fold_selections)

    report_failures = [] if args.skip_reports else generate_reports(args)
    failures.extend(report_failures)

    if failures:
        print("[run_all_models] Completed with failures:")
        for failure in failures:
            print(f"  - {failure['model']} (returncode={failure['returncode']})")
        raise SystemExit(1)
    command_status["value"] = "ok"


def run_sequential(
    config_paths: list[Path],
    args: argparse.Namespace,
    *,
    fold_selections: list[FoldSelection],
) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    plan = build_job_plan(config_paths, fold_selections=fold_selections)
    for item in plan:
        command = build_command(item.config_path, args, device=None, fold=item.fold)
        print(f"[run_all_models] Running {format_job_progress(item)}")
        result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            failures.append(
                {
                    "model": f"{item.config_path.parent.name}:{item.fold.label}",
                    "returncode": result.returncode,
                }
            )
            print(
                f"[run_all_models] Model failed: {item.config_path.parent.name} "
                f"fold={item.fold.label} (returncode={result.returncode})"
            )
    return failures


def run_parallel(
    config_paths: list[Path],
    args: argparse.Namespace,
    *,
    fold_selections: list[FoldSelection],
) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    pending = deque(build_job_plan(config_paths, fold_selections=fold_selections))
    active: dict[int, RunningJob] = {}
    devices = list(args.resolved_devices or [])

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            item = pending.popleft()
            command = build_command(item.config_path, args, device=device, fold=item.fold)
            env = build_subprocess_env(device=device)
            started_at = time.time()
            print(f"[run_all_models] Launching {format_job_progress(item)} on GPU {device}")
            process = subprocess.Popen(command, env=env, cwd=REPO_ROOT)
            active[device] = RunningJob(device=device, plan_item=item, process=process, started_at=started_at)

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            item = job.plan_item
            print(
                f"[run_all_models] Finished {format_job_progress(item)} on GPU {device} "
                f"(returncode={returncode}, duration={time.time() - job.started_at:.1f}s)"
            )
            if returncode != 0:
                failures.append(
                    {
                        "model": f"{item.config_path.parent.name}:{item.fold.label}",
                        "returncode": returncode,
                    }
                )
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        if active:
            time.sleep(2)

    return failures


def build_job_plan(config_paths: list[Path], *, fold_selections: list[FoldSelection]) -> list[JobPlanItem]:
    total_models = len(config_paths)
    total_folds = len(fold_selections)
    total_jobs = total_models * total_folds
    items: list[JobPlanItem] = []
    job_index = 0
    for fold_index, fold in enumerate(fold_selections, start=1):
        for model_index, config_path in enumerate(config_paths, start=1):
            job_index += 1
            items.append(
                JobPlanItem(
                    job_index=job_index,
                    total_jobs=total_jobs,
                    model_index=model_index,
                    total_models=total_models,
                    fold_index=fold_index,
                    total_folds=total_folds,
                    config_path=config_path,
                    fold=fold,
                )
            )
    return items


def format_job_progress(item: JobPlanItem) -> str:
    return (
        f"job={item.job_index}/{item.total_jobs} "
        f"model={item.model_index}/{item.total_models} {item.config_path.parent.name} "
        f"fold={item.fold_index}/{item.total_folds} {item.fold.label}"
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


def build_command(
    config_path: Path,
    args: argparse.Namespace,
    *,
    device: int | None,
    fold: FoldSelection | None = None,
) -> list[str]:
    fold = fold or fold_from_args(args)
    split_protocol = getattr(args, "split_protocol", "nested_cv")
    nested_split_csv = getattr(
        args,
        "nested_split_csv",
        "data/splits/isic2024_official_train_nested_5x4_seed42.csv",
    )
    outer_fold = fold.outer_fold if fold.outer_fold is not None else getattr(args, "outer_fold", 0)
    inner_fold = fold.inner_fold if fold.inner_fold is not None else getattr(args, "inner_fold", 0)
    cv_fold = fold.cv_fold if fold.cv_fold is not None else getattr(args, "cv_fold", 0)
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_image_baseline",
        "--config",
        str(config_path),
        "--dataset-root",
        str(resolve_repo_path(args.dataset_root)),
        "--output-root",
        str(command_output_root(args, fold)),
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
        str(cv_fold),
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
    if getattr(args, "batch_size_override", None) is not None:
        command.extend(["--batch-size-override", str(args.batch_size_override)])
    if args.disable_pretrained:
        command.append("--disable-pretrained")
    return command


def fold_from_args(args: argparse.Namespace) -> FoldSelection:
    if getattr(args, "split_protocol", "nested_cv") == "nested_cv":
        return FoldSelection(
            outer_fold=int(getattr(args, "outer_fold", 0)),
            inner_fold=int(getattr(args, "inner_fold", 0)),
        )
    return FoldSelection(cv_fold=int(getattr(args, "cv_fold", 0)))


def command_output_root(args: argparse.Namespace, fold: FoldSelection) -> Path:
    output_root = resolve_repo_path(args.output_root)
    if getattr(args, "all_folds", False):
        return output_root / fold.label
    return output_root


def resolve_fold_selections(args: argparse.Namespace) -> list[FoldSelection]:
    if not getattr(args, "all_folds", False):
        return [fold_from_args(args)]
    if getattr(args, "split_protocol", "nested_cv") == "nested_cv":
        return resolve_nested_fold_selections(args.nested_split_csv)
    return resolve_legacy_cv_fold_selections(args.cv_split_csv)


def resolve_nested_fold_selections(nested_split_csv: str | Path) -> list[FoldSelection]:
    import pandas as pd

    nested_path = resolve_repo_path(nested_split_csv)
    if not nested_path.exists():
        raise FileNotFoundError(f"Nested split CSV not found: {nested_path}")
    split_frame = pd.read_csv(nested_path, usecols=["outer_fold", "inner_fold"], low_memory=False)
    fold_pairs = (
        split_frame[["outer_fold", "inner_fold"]]
        .drop_duplicates()
        .sort_values(["outer_fold", "inner_fold"])
    )
    if fold_pairs.empty:
        raise RuntimeError(f"Nested split CSV has no fold pairs: {nested_path}")
    return [
        FoldSelection(outer_fold=int(row.outer_fold), inner_fold=int(row.inner_fold))
        for row in fold_pairs.itertuples(index=False)
    ]


def resolve_legacy_cv_fold_selections(cv_split_csv: str | Path) -> list[FoldSelection]:
    import pandas as pd

    cv_path = resolve_repo_path(cv_split_csv)
    if not cv_path.exists():
        raise FileNotFoundError(f"CV split CSV not found: {cv_path}")
    cv_frame = pd.read_csv(cv_path, usecols=["cv_validation_fold"], low_memory=False)
    cv_folds = sorted(int(value) for value in cv_frame["cv_validation_fold"].dropna().unique().tolist())
    if not cv_folds:
        raise RuntimeError(f"CV split CSV has no cv_validation_fold values: {cv_path}")
    return [FoldSelection(cv_fold=cv_fold) for cv_fold in cv_folds]


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
