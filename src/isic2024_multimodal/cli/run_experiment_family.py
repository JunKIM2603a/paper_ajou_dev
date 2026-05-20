from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from isic2024_multimodal.experiments.dataset_specs import dataset_fingerprint, load_dataset_spec
from isic2024_multimodal.experiments.families import (
    EXPERIMENT_FAMILIES,
    FamilyPaths,
    load_suite_config,
    make_run_group_id,
    reset_family_outputs,
    resolve_family_paths,
    write_json,
)
from isic2024_multimodal.experiments.registry import write_family_selection
from isic2024_multimodal.utils.progress import format_progress_duration
from isic2024_multimodal.utils.runtime_env import (
    DEFAULT_MLFLOW_FILE_TRACKING_URI,
    DEFAULT_MLFLOW_SQLITE_TRACKING_URI,
    ensure_expected_conda_env,
    load_project_env,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an independent ISIC2024 experiment family.")
    parser.add_argument("--family", required=True, choices=sorted(EXPERIMENT_FAMILIES))
    parser.add_argument("--config", default=None, help="Suite config JSON. Defaults to experiments/configs/suites/<family>.json.")
    parser.add_argument("--dataset-spec", default=None, help="Dataset spec JSON override.")
    parser.add_argument("--run-group-id", default=None, help="Reusable run group id. Defaults to timestamp.")
    parser.add_argument("--tracking-uri", default=get_repo_default_mlflow_tracking_uri())
    parser.add_argument("--devices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu"],
        default="auto",
        help="Device policy for family runners. auto prefers CUDA and falls back to CPU; cpu forces CPU.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reset-family-output", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_project_env()
    ensure_expected_conda_env()
    args = parse_args()
    args.run_group_id = args.run_group_id or make_run_group_id(args.family)
    paths = resolve_family_paths(
        family=args.family,
        run_group_id=args.run_group_id,
        repo_root=REPO_ROOT,
        smoke=args.smoke,
    )
    if args.reset_family_output:
        reset_targets = reset_family_outputs(paths, dry_run=args.dry_run)
        print(json.dumps({"reset_targets": [str(path) for path in reset_targets], "dry_run": args.dry_run}, indent=2))
        return
    if args.resume and paths.status_path.exists():
        status = json.loads(paths.status_path.read_text(encoding="utf-8"))
        if status.get("status") == "ok":
            print(json.dumps({"status": "skipped_existing_ok", "status_path": str(paths.status_path)}, indent=2))
            return

    suite_path = Path(args.config or default_suite_path(args.family))
    suite = load_suite_config(suite_path, repo_root=REPO_ROOT)
    dataset_spec_path = args.dataset_spec or suite.get("dataset_spec")
    if not dataset_spec_path:
        raise RuntimeError(f"Suite config is missing `dataset_spec`: {suite_path}")
    dataset_spec = load_dataset_spec(dataset_spec_path, repo_root=REPO_ROOT)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.table_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        args=args,
        suite=suite,
        suite_path=suite_path,
        dataset_spec=dataset_spec,
        paths=paths,
    )
    write_json(paths.run_manifest_path, manifest)
    preflight = build_preflight_summary(manifest)
    write_json(paths.preflight_path, preflight)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    command = build_family_command(
        family=args.family,
        suite=suite,
        dataset_spec=dataset_spec,
        paths=paths,
        args=args,
    )
    status = {
        "family": args.family,
        "run_group_id": args.run_group_id,
        "dataset_id": dataset_spec.dataset_id,
        "command": command,
        "started_at": current_timestamp(),
        "status": "dry_run" if args.dry_run else "running",
    }
    write_json(paths.status_path, status)
    print(json.dumps({"command": command, "dry_run": args.dry_run}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    started = time.time()
    log_event(
        f"Start family={args.family} run_group_id={args.run_group_id} "
        f"dataset_id={dataset_spec.dataset_id} output_root={paths.output_root}"
    )
    result = subprocess.run(command, cwd=REPO_ROOT, env=build_subprocess_env(), check=False)
    status.update(
        {
            "ended_at": current_timestamp(),
            "duration_seconds": time.time() - started,
            "returncode": result.returncode,
            "status": "ok" if result.returncode == 0 else "failed",
        }
    )
    selection = None
    if result.returncode == 0:
        selection = write_family_selection(
            family=args.family,
            run_group_id=args.run_group_id,
            dataset_id=dataset_spec.dataset_id,
            output_root=paths.output_root,
            table_root=paths.table_root,
            config_path=suite_path,
            dataset_spec_path=dataset_spec.path,
            repo_root=REPO_ROOT,
        )
        status["selection"] = selection
    write_json(paths.status_path, status)
    log_event(
        f"Finished family={args.family} run_group_id={args.run_group_id} "
        f"returncode={result.returncode} status_path={paths.status_path} "
        f"duration={format_progress_duration(status['duration_seconds'])}"
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def build_run_manifest(
    *,
    args: argparse.Namespace,
    suite: dict[str, Any],
    suite_path: Path,
    dataset_spec,
    paths: FamilyPaths,
) -> dict[str, Any]:
    return {
        "family": args.family,
        "run_group_id": args.run_group_id,
        "smoke": bool(args.smoke),
        "suite_config_path": str(suite_path),
        "dataset": dataset_spec.to_manifest(),
        "raw_dataset_fingerprint": dataset_fingerprint(dataset_spec),
        "output_root": str(paths.output_root),
        "table_root": str(paths.table_root),
        "tracking_uri": args.tracking_uri,
        "experiment_name": suite.get("experiment_name", EXPERIMENT_FAMILIES[args.family]),
        "devices": args.devices,
        "device_policy": args.device_policy,
        "created_at": current_timestamp(),
    }


def log_event(message: str) -> None:
    print(f"[{current_timestamp()}] [run_experiment_family] {message}", flush=True)


def build_preflight_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = manifest["dataset"]
    return {
        "status": "ok",
        "family": manifest["family"],
        "run_group_id": manifest["run_group_id"],
        "dataset_id": dataset["dataset_id"],
        "dataset_spec_path": dataset["dataset_spec_path"],
        "split_protocol": dataset["split_protocol"],
        "split_source": "nested_cv_split_csv" if dataset["split_protocol"] == "nested_cv" else "locked_split_csv",
        "nested_split_csv": dataset["nested_split_csv"],
        "outer_fold": dataset["outer_fold"],
        "inner_fold": dataset["inner_fold"],
        "holdout_split_csv": dataset["holdout_split_csv"],
        "cv_split_csv": dataset["cv_split_csv"],
        "device_policy": manifest.get("device_policy", "auto"),
        "requested_devices": manifest.get("devices"),
        "output_root": manifest["output_root"],
        "table_root": manifest["table_root"],
    }


def build_family_command(
    *,
    family: str,
    suite: dict[str, Any],
    dataset_spec,
    paths: FamilyPaths,
    args: argparse.Namespace,
) -> list[str]:
    if family == "tabular_baselines":
        return build_tabular_command(suite=suite, dataset_spec=dataset_spec, paths=paths, args=args)
    if family == "image_baselines":
        return build_image_command(suite=suite, dataset_spec=dataset_spec, paths=paths, args=args)
    if family in {"multimodal_baselines", "final_paper_model"}:
        return build_multimodal_command(family=family, suite=suite, dataset_spec=dataset_spec, paths=paths, args=args)
    raise ValueError(f"Unknown family: {family}")


def build_tabular_command(*, suite: dict[str, Any], dataset_spec, paths: FamilyPaths, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_all_tabular_models",
        "--dataset-root",
        str(dataset_spec.dataset_root),
        "--feature-set-json",
        str(dataset_spec.feature_set_json),
        "--output-root",
        str(paths.output_root),
        "--tracking-uri",
        args.tracking_uri,
        "--experiment-name",
        suite.get("experiment_name", EXPERIMENT_FAMILIES["tabular_baselines"]),
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        dataset_spec.dataset_id,
        "--dataset-spec",
        str(dataset_spec.path),
        "--model-family",
        "tabular_baselines",
        "--split-protocol",
        dataset_spec.split_protocol,
        "--nested-split-csv",
        str(dataset_spec.nested_split_csv),
        "--outer-fold",
        str(dataset_spec.outer_fold),
        "--inner-fold",
        str(dataset_spec.inner_fold),
        "--holdout-split-csv",
        str(dataset_spec.holdout_split_csv),
        "--cv-split-csv",
        str(dataset_spec.cv_split_csv),
        "--cv-fold",
        str(dataset_spec.cv_fold),
        "--leaderboard-output",
        str(paths.table_root / "mlflow_leaderboard.csv"),
        "--html-report-output",
        str(paths.table_root / "mlflow_report.html"),
        "--device-policy",
        getattr(args, "device_policy", "auto"),
    ]
    append_list(command, "--models", suite.get("models"))
    append_list(command, "--feature-sets", suite.get("feature_sets", dataset_spec.feature_sets))
    append_devices(command, args.devices)
    append_smoke_caps(command, suite=suite, args=args, prefix="rows")
    if args.skip_reports:
        command.append("--skip-reports")
    return command


def build_image_command(*, suite: dict[str, Any], dataset_spec, paths: FamilyPaths, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_all_image_models",
        "--config-root",
        str(resolve_repo_path(suite.get("config_root", "experiments/configs/image_baselines"))),
        "--dataset-root",
        str(dataset_spec.dataset_root),
        "--output-root",
        str(paths.output_root),
        "--tracking-uri",
        args.tracking_uri,
        "--experiment-name",
        suite.get("experiment_name", EXPERIMENT_FAMILIES["image_baselines"]),
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        dataset_spec.dataset_id,
        "--dataset-spec",
        str(dataset_spec.path),
        "--model-family",
        "image_baselines",
        "--split-protocol",
        dataset_spec.split_protocol,
        "--nested-split-csv",
        str(dataset_spec.nested_split_csv),
        "--outer-fold",
        str(dataset_spec.outer_fold),
        "--inner-fold",
        str(dataset_spec.inner_fold),
        "--holdout-split-csv",
        str(dataset_spec.holdout_split_csv),
        "--cv-split-csv",
        str(dataset_spec.cv_split_csv),
        "--cv-fold",
        str(dataset_spec.cv_fold),
        "--leaderboard-output",
        str(paths.table_root / "mlflow_leaderboard.csv"),
        "--html-report-output",
        str(paths.table_root / "mlflow_report.html"),
        "--device-policy",
        getattr(args, "device_policy", "auto"),
    ]
    append_list(command, "--models", suite.get("models"))
    append_list(command, "--exclude-models", suite.get("exclude_models"))
    append_devices(command, args.devices)
    append_smoke_caps(command, suite=suite, args=args, prefix="samples")
    if args.skip_reports:
        command.append("--skip-reports")
    if suite.get("disable_pretrained"):
        command.append("--disable-pretrained")
    return command


def build_multimodal_command(
    *,
    family: str,
    suite: dict[str, Any],
    dataset_spec,
    paths: FamilyPaths,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_multimodal_experiment",
        "--dataset-root",
        str(dataset_spec.dataset_root),
        "--config",
        str(resolve_repo_path(suite.get("config", "experiments/configs/multimodal/default.json"))),
        "--output-root",
        str(paths.output_root),
        "--tracking-uri",
        args.tracking_uri,
        "--experiment-name",
        suite.get("experiment_name", EXPERIMENT_FAMILIES[family]),
        "--run-group-id",
        args.run_group_id,
        "--dataset-id",
        dataset_spec.dataset_id,
        "--dataset-spec",
        str(dataset_spec.path),
        "--model-family",
        family,
        "--split-protocol",
        dataset_spec.split_protocol,
        "--nested-split-csv",
        str(dataset_spec.nested_split_csv),
        "--outer-fold",
        str(dataset_spec.outer_fold),
        "--inner-fold",
        str(dataset_spec.inner_fold),
        "--device",
        getattr(args, "device_policy", "auto"),
    ]
    return command


def append_list(command: list[str], flag: str, values: list[str] | None) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def append_devices(command: list[str], devices: list[int] | None) -> None:
    if devices:
        command.append("--devices")
        command.extend(str(device) for device in devices)


def append_smoke_caps(command: list[str], *, suite: dict[str, Any], args: argparse.Namespace, prefix: str) -> None:
    values = suite.get("smoke" if args.smoke else "limits", {})
    if prefix == "rows":
        mapping = {
            "max_train": "--max-train-rows",
            "max_val": "--max-val-rows",
            "max_test": "--max-test-rows",
        }
    else:
        mapping = {
            "max_train": "--max-train-samples",
            "max_val": "--max-val-samples",
            "max_test": "--max-test-samples",
            "epochs": "--epochs-override",
            "max_trials": "--max-trials",
        }
    for key, flag in mapping.items():
        if key in values and values[key] is not None:
            command.extend([flag, str(values[key])])


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(SRC_ROOT)
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    if current_pythonpath:
        entries = current_pythonpath.split(os.pathsep)
        if src_path not in entries:
            env["PYTHONPATH"] = os.pathsep.join([src_path, *entries])
    else:
        env["PYTHONPATH"] = src_path
    return env


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def default_suite_path(family: str) -> Path:
    filename = "final_paper_model_ablation.json" if family == "final_paper_model" else f"{family}.json"
    return REPO_ROOT / "experiments" / "configs" / "suites" / filename


def get_repo_default_mlflow_tracking_uri() -> str:
    if (REPO_ROOT / "experiments/logs/mlflow.db").exists():
        return DEFAULT_MLFLOW_SQLITE_TRACKING_URI
    return DEFAULT_MLFLOW_FILE_TRACKING_URI


if __name__ == "__main__":
    main()
