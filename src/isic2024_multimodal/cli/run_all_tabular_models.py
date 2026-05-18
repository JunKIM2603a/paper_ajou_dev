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
DEFAULT_MODELS = [
    "logistic_regression",
    "svm",
    "mlp",
    "xgboost",
    "catboost",
    "lightgbm",
    "ft_transformer",
    "ft_transformer_external",
]
DEFAULT_FEATURE_SETS = ["strict_base", "strict_fe", "strict_main_input"]


def make_run_group_id(prefix: str = "tabular_all") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


@dataclass
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


@dataclass
class RunningJob:
    device: int
    model_name: str
    fold: FoldSelection
    process: subprocess.Popen
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular baseline models in parallel across multiple GPUs.")
    parser.add_argument("--dataset-root", default="data/raw/isic_2024_challenge")
    parser.add_argument("--eda-dir", default="experiments/evidence/eda/isic_2024")
    parser.add_argument("--feature-set-json", default="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json")
    parser.add_argument("--output-root", default="experiments/outputs/tabular_baselines")
    parser.add_argument("--tracking-uri", default=get_repo_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Baselines")
    parser.add_argument("--run-group-id", default=None, help="Optional MLflow run group tag. Defaults to a timestamp.")
    parser.add_argument("--dataset-id", default=None, help="Versioned dataset id for registry/report filtering.")
    parser.add_argument("--dataset-spec", default=None, help="Dataset spec JSON path used for this run.")
    parser.add_argument("--model-family", default="tabular_baselines", help="Experiment family tag.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-protocol", choices=["nested_cv", "legacy_holdout"], default="nested_cv")
    parser.add_argument("--nested-split-csv", default="data/splits/isic2024_official_train_nested_5x4_seed42.csv")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument(
        "--all-folds",
        "--all-nested-folds",
        action="store_true",
        dest="all_folds",
        help=(
            "Run every fold found in the split artifact. For nested_cv this runs every "
            "(outer_fold, inner_fold) pair; for legacy_holdout this runs every cv_fold."
        ),
    )
    parser.add_argument("--holdout-split-csv", default="data/splits/isic2024_train_validation_test_split_seed42.csv")
    parser.add_argument("--cv-split-csv", default="data/splits/isic2024_train_validation_5fold_seed42.csv")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--devices", nargs="*", type=int, default=None, help="Visible GPU indices to run in parallel.")
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu"],
        default="auto",
        help="Device policy for subprocesses. auto prefers CUDA and falls back to CPU; cpu forces CPU.",
    )
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
    command_start = time.time()
    load_project_env()
    args = parse_args()
    args.run_group_id = args.run_group_id or make_run_group_id()
    command_status = {"value": "failed"}

    def log_command_end() -> None:
        log_event(
            f"End status={command_status['value']} "
            f"run_group_id={args.run_group_id} duration={format_duration(time.time() - command_start)}"
        )

    atexit.register(log_command_end)
    ensure_expected_conda_env()
    log_event(
        "Start "
        f"run_group_id={args.run_group_id} models={','.join(args.models)} "
        f"feature_sets={','.join(args.feature_sets)} requested_devices={args.devices or 'auto'} "
        f"device_policy={args.device_policy}"
    )

    device_resolution = resolve_device_list(args.devices, device_policy=args.device_policy)
    args.resolved_devices = device_resolution.resolved_devices
    args.device_fallback_reason = device_resolution.fallback_reason
    log_event(
        f"Resolved devices={args.resolved_devices or 'cpu'} "
        f"cuda_available={device_resolution.cuda_available} "
        f"visible_device_count={device_resolution.visible_device_count} "
        f"fallback={args.device_fallback_reason or 'none'}"
    )
    fold_selections = resolve_fold_selections(args)
    log_event(
        "Resolved folds="
        f"{','.join(fold.label for fold in fold_selections)} "
        f"count={len(fold_selections)}"
    )
    preflight_device = args.resolved_devices[0] if args.resolved_devices else None
    preflight_failures = run_preflights(args, device=preflight_device, fold_selections=fold_selections)
    if preflight_failures:
        log_event("Preflight failed; no model jobs were launched.")
        raise SystemExit(1)

    if args.resolved_devices:
        failures = run_parallel(args, fold_selections=fold_selections)
    else:
        failures = run_sequential(args, fold_selections=fold_selections)

    report_failures = [] if args.skip_reports else generate_reports(args)
    failures.extend(report_failures)

    if failures:
        log_event(f"Completed with {len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure['model']} (returncode={failure['returncode']})")
        raise SystemExit(1)
    command_status["value"] = "ok"


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def log_event(message: str) -> None:
    print(f"[{current_timestamp()}] [run_all_tabular_models] {message}", flush=True)


def run_sequential(args: argparse.Namespace, *, fold_selections: list[FoldSelection]) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    env = build_subprocess_env()
    for fold in fold_selections:
        for model_name in args.models:
            command = build_command(model_name, args, device=None, fold=fold)
            model_start = time.time()
            log_event(f"Start model={model_name} fold={fold.label} device=cpu")
            result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
            log_event(
                f"Finished model={model_name} fold={fold.label} device=cpu returncode={result.returncode} "
                f"duration={format_duration(time.time() - model_start)}"
            )
            if result.returncode != 0:
                failures.append({"model": f"{model_name}:{fold.label}", "returncode": result.returncode})
    return failures


def run_parallel(args: argparse.Namespace, *, fold_selections: list[FoldSelection]) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    pending = deque((fold, model_name) for fold in fold_selections for model_name in args.models)
    active: dict[int, RunningJob] = {}
    devices = list(args.resolved_devices or [])

    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            fold, model_name = pending.popleft()
            command = build_command(model_name, args, device=device, fold=fold)
            env = build_subprocess_env(device=device)
            started_at = time.time()
            log_event(f"Start model={model_name} fold={fold.label} gpu={device}")
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
            active[device] = RunningJob(
                device=device,
                model_name=model_name,
                fold=fold,
                process=process,
                started_at=started_at,
            )

        completed_devices: list[int] = []
        for device, job in active.items():
            returncode = job.process.poll()
            if returncode is None:
                continue
            log_event(
                f"Finished model={job.model_name} fold={job.fold.label} gpu={device} returncode={returncode} "
                f"duration={format_duration(time.time() - job.started_at)}"
            )
            if returncode != 0:
                failures.append({"model": f"{job.model_name}:{job.fold.label}", "returncode": returncode})
            completed_devices.append(device)

        for device in completed_devices:
            del active[device]

        if active:
            time.sleep(2)

    return failures


def run_preflights(
    args: argparse.Namespace,
    *,
    device: int | None,
    fold_selections: list[FoldSelection],
) -> list[dict[str, str | int]]:
    failures: list[dict[str, str | int]] = []
    for fold in fold_selections:
        failures.extend(run_preflight(args, device=device, fold=fold))
    return failures


def run_preflight(
    args: argparse.Namespace,
    *,
    device: int | None,
    fold: FoldSelection | None = None,
) -> list[dict[str, str | int]]:
    fold = fold or fold_from_args(args)
    command = build_preflight_command(args, device=device, fold=fold)
    env = build_subprocess_env(device=device)
    preflight_start = time.time()
    log_event(f"Start preflight fold={fold.label}")
    result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
    log_event(
        f"Finished preflight fold={fold.label} returncode={result.returncode} "
        f"duration={format_duration(time.time() - preflight_start)}"
    )
    if result.returncode != 0:
        return [{"model": f"preflight:{fold.label}", "returncode": result.returncode}]
    return []


def build_preflight_command(
    args: argparse.Namespace,
    *,
    device: int | None,
    fold: FoldSelection | None = None,
) -> list[str]:
    fold = fold or fold_from_args(args)
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
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        getattr(args, "dataset_id", None) or "",
        "--dataset-spec",
        getattr(args, "dataset_spec", None) or "",
        "--model-family",
        getattr(args, "model_family", "tabular_baselines"),
        "--output-root",
        str(command_output_root(args, fold)),
        "--tracking-uri",
        args.tracking_uri,
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--split-protocol",
        args.split_protocol,
        "--nested-split-csv",
        str(resolve_repo_path(args.nested_split_csv)),
        "--outer-fold",
        str(fold.outer_fold if fold.outer_fold is not None else args.outer_fold),
        "--inner-fold",
        str(fold.inner_fold if fold.inner_fold is not None else args.inner_fold),
        "--cv-fold",
        str(fold.cv_fold if fold.cv_fold is not None else args.cv_fold),
        "--holdout-split-csv",
        str(resolve_repo_path(args.holdout_split_csv)),
        "--cv-split-csv",
        str(resolve_repo_path(args.cv_split_csv)),
        "--preflight-only",
    ]
    if args.models:
        command.append("--models")
        command.extend(args.models)
    if args.feature_sets:
        command.append("--feature-sets")
        command.extend(args.feature_sets)
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_val_rows is not None:
        command.extend(["--max-val-rows", str(args.max_val_rows)])
    if args.max_test_rows is not None:
        command.extend(["--max-test-rows", str(args.max_test_rows)])
    device_arg = command_device_arg(args, device)
    if device_arg is not None:
        command.extend(["--device", device_arg])
    return command


def build_command(
    model_name: str,
    args: argparse.Namespace,
    *,
    device: int | None,
    fold: FoldSelection | None = None,
) -> list[str]:
    fold = fold or fold_from_args(args)
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
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        getattr(args, "dataset_id", None) or "",
        "--dataset-spec",
        getattr(args, "dataset_spec", None) or "",
        "--model-family",
        getattr(args, "model_family", "tabular_baselines"),
        "--output-root",
        str(command_output_root(args, fold)),
        "--tracking-uri",
        args.tracking_uri,
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--split-protocol",
        args.split_protocol,
        "--nested-split-csv",
        str(resolve_repo_path(args.nested_split_csv)),
        "--outer-fold",
        str(fold.outer_fold if fold.outer_fold is not None else args.outer_fold),
        "--inner-fold",
        str(fold.inner_fold if fold.inner_fold is not None else args.inner_fold),
        "--cv-fold",
        str(fold.cv_fold if fold.cv_fold is not None else args.cv_fold),
        "--holdout-split-csv",
        str(resolve_repo_path(args.holdout_split_csv)),
        "--cv-split-csv",
        str(resolve_repo_path(args.cv_split_csv)),
        "--models",
        model_name,
    ]
    if args.feature_sets:
        command.append("--feature-sets")
        command.extend(args.feature_sets)
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_val_rows is not None:
        command.extend(["--max-val-rows", str(args.max_val_rows)])
    if args.max_test_rows is not None:
        command.extend(["--max-test-rows", str(args.max_test_rows)])
    device_arg = command_device_arg(args, device)
    if device_arg is not None:
        command.extend(["--device", device_arg])
    return command


def command_device_arg(args: argparse.Namespace, device: int | None) -> str | None:
    if device is not None:
        return "cuda"
    if getattr(args, "device_policy", "auto") == "cpu":
        return "cpu"
    if getattr(args, "devices", None) and not getattr(args, "resolved_devices", []):
        return "cpu"
    return None


def fold_from_args(args: argparse.Namespace) -> FoldSelection:
    if getattr(args, "split_protocol", "nested_cv") == "nested_cv":
        return FoldSelection(outer_fold=int(args.outer_fold), inner_fold=int(args.inner_fold))
    return FoldSelection(cv_fold=int(args.cv_fold))


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
                "--run-group-id",
                args.run_group_id,
                "--dataset-id",
                getattr(args, "dataset_id", None) or "",
                "--model-family",
                getattr(args, "model_family", "tabular_baselines"),
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
                "--run-group-id",
                args.run_group_id,
                "--dataset-id",
                getattr(args, "dataset_id", None) or "",
                "--model-family",
                getattr(args, "model_family", "tabular_baselines"),
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
        report_start = time.time()
        log_event(f"Start report={spec['name']} output={spec['output']}")
        result = subprocess.run(spec["command"], check=False, cwd=REPO_ROOT, env=env)
        log_event(
            f"Finished report={spec['name']} returncode={result.returncode} "
            f"duration={format_duration(time.time() - report_start)}"
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
    if Path(REPO_ROOT / "experiments/logs/mlflow.db").exists():
        return DEFAULT_MLFLOW_SQLITE_TRACKING_URI
    return DEFAULT_MLFLOW_FILE_TRACKING_URI


if __name__ == "__main__":
    main()
