from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from isic2024_multimodal.models.image.checkpoint_preflight import preflight_image_model_config
from isic2024_multimodal.utils.config_utils import load_json, sanitize_run_name


DEFAULT_SUITE = "experiments/configs/suites/image_baselines.json"
DEFAULT_OUTPUT_ROOT = "experiments/outputs/image_baselines"
DEFAULT_STATUS_ROOT = "experiments/tables/image_baselines/status"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report one-by-one image baseline test status.")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-root", default=DEFAULT_STATUS_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write CSV/JSON status files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    records = build_status_records(
        suite_path=repo_root / args.suite,
        output_root=repo_root / args.output_root,
        repo_root=repo_root,
    )
    print_status_table(records)
    if not args.no_write:
        output_paths = write_status_reports(records, repo_root / args.status_root)
        print(json.dumps({key: str(value) for key, value in output_paths.items()}, indent=2))


def build_status_records(
    *,
    suite_path: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    suite_path = Path(suite_path)
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    suite = load_json(suite_path)
    config_root = resolve_repo_path(suite.get("config_root", "experiments/configs/image_baselines"), repo_root=repo_root)
    records = []
    for model_key in suite.get("models", []):
        config_path = config_root / model_key / "config.json"
        config = load_json(config_path)
        model_config = config["model"]
        display_name = str(model_config.get("display_name", model_key))
        model_output_root = output_root / sanitize_run_name(display_name)
        checkpoint_status = checkpoint_status_for_model(model_config, repo_root=repo_root)
        artifact_status = artifact_status_for_model(model_output_root)
        status = classify_model_status(checkpoint_status, artifact_status)
        records.append(
            {
                "model_key": model_key,
                "model_name": display_name,
                "status": status,
                "checkpoint_status": checkpoint_status["status"],
                "checkpoint_path": checkpoint_status.get("path", ""),
                "checkpoint_notes": checkpoint_status.get("notes", ""),
                "latest_summary_path": artifact_status.get("summary_path", ""),
                "latest_error_path": artifact_status.get("error_path", ""),
                "latest_run_status_path": artifact_status.get("run_status_path", ""),
                "duration_seconds": artifact_status.get("duration_seconds", ""),
                "best_validation_metric": artifact_status.get("best_validation_metric", ""),
                "failure_type": artifact_status.get("failure_type", ""),
                "failure_message": artifact_status.get("failure_message", ""),
            }
        )
    return records


def checkpoint_status_for_model(model_config: dict[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    checkpoint_path = model_config.get("checkpoint_path")
    if checkpoint_path:
        resolved = resolve_repo_path(checkpoint_path, repo_root=repo_root)
        if resolved.exists():
            try:
                preflight_image_model_config(model_config, repo_root=repo_root)
            except Exception as exc:
                return {
                    "status": "preflight_failed",
                    "path": str(resolved),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            return {"status": "exists", "path": str(resolved), "notes": "local checkpoint exists and preflight passed"}
        return {"status": "missing", "path": str(resolved), "notes": "no manual checkpoint downloader is used"}

    return {"status": "hub/cache", "path": "", "notes": "uses library/hub cache; no manual checkpoint path"}


def resolve_repo_path(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(repo_root) / value


def artifact_status_for_model(model_output_root: Path) -> dict[str, Any]:
    if not model_output_root.exists():
        return {}

    run_statuses = sorted(model_output_root.glob("**/run_status.json"), key=lambda path: path.stat().st_mtime)
    summaries = sorted(model_output_root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime)
    errors = sorted(model_output_root.glob("**/error.txt"), key=lambda path: path.stat().st_mtime)

    latest_run_status = load_json(run_statuses[-1]) if run_statuses else {}
    latest_summary = load_json(summaries[-1]) if summaries else {}
    latest_error = errors[-1] if errors else None

    payload: dict[str, Any] = {}
    if run_statuses:
        payload["run_status_path"] = str(run_statuses[-1])
        payload.update({key: latest_run_status.get(key) for key in ["failure_type", "failure_message"]})
    if summaries:
        payload["summary_path"] = str(summaries[-1])
        payload["duration_seconds"] = latest_summary.get("duration_seconds")
        payload["best_validation_metric"] = latest_summary.get("best_validation_metric")
        payload["is_smoke"] = is_smoke_summary(summaries[-1], latest_run_status)
    if latest_error is not None:
        payload["error_path"] = str(latest_error)
    if latest_run_status.get("status") == "running":
        payload["artifact_status"] = "running"
    elif latest_error is not None and (not summaries or latest_error.stat().st_mtime > summaries[-1].stat().st_mtime):
        payload["artifact_status"] = "failed"
    elif summaries:
        payload["artifact_status"] = "smoke_passed" if payload.get("is_smoke") else "completed"
    return payload


def is_smoke_summary(summary_path: Path, run_status: dict[str, Any]) -> bool:
    hyperparameters = run_status.get("hyperparameters") if isinstance(run_status, dict) else None
    if isinstance(hyperparameters, dict) and int(hyperparameters.get("epochs", 0) or 0) <= 1:
        return True
    summary = load_json(summary_path)
    history_path = summary.get("history_path")
    if history_path and Path(history_path).exists():
        with Path(history_path).open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return len(rows) <= 1
    return False


def classify_model_status(checkpoint_status: dict[str, Any], artifact_status: dict[str, Any]) -> str:
    artifact = artifact_status.get("artifact_status")
    if artifact in {"running", "failed", "smoke_passed", "completed"}:
        return str(artifact)
    if checkpoint_status.get("status") in {"missing", "downloadable"}:
        return "checkpoint_missing"
    if checkpoint_status.get("status") == "preflight_failed":
        return "preflight_failed"
    return "not_started"


def write_status_reports(records: list[dict[str, Any]], status_root: Path) -> dict[str, Path]:
    status_root.mkdir(parents=True, exist_ok=True)
    csv_path = status_root / "image_baseline_status.csv"
    json_path = status_root / "image_baseline_status.json"
    columns = [
        "model_key",
        "model_name",
        "status",
        "checkpoint_status",
        "checkpoint_path",
        "duration_seconds",
        "best_validation_metric",
        "failure_type",
        "latest_summary_path",
        "latest_error_path",
        "latest_run_status_path",
        "checkpoint_notes",
        "failure_message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def print_status_table(records: list[dict[str, Any]]) -> None:
    columns = ["model_key", "status", "checkpoint_status", "duration_seconds", "best_validation_metric"]
    widths = {
        column: max(len(column), *(len(str(record.get(column, ""))) for record in records))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for record in records:
        print("  ".join(str(record.get(column, "")).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
