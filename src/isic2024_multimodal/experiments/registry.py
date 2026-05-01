from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from isic2024_multimodal.evaluation.metrics import PRIMARY_PAUC_METRIC


SELECTION_FILENAMES = {
    "tabular_baselines": "best_tabular_by_run_group.json",
    "image_baselines": "best_image_by_run_group.json",
    "multimodal_baselines": "best_multimodal_by_run_group.json",
    "final_paper_model": "best_final_paper_model_by_run_group.json",
}


def read_selection_registry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selection_registry_path(family: str, *, repo_root: str | Path | None = None) -> Path:
    repo_root = Path(repo_root or Path.cwd()).resolve()
    filename = SELECTION_FILENAMES.get(family)
    if filename is None:
        raise ValueError(f"Unsupported selection registry family: {family}")
    return repo_root / "experiments" / "registry" / "selections" / filename


def write_family_selection(
    *,
    family: str,
    run_group_id: str,
    dataset_id: str,
    output_root: str | Path,
    table_root: str | Path,
    config_path: str | Path,
    dataset_spec_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any] | None:
    output_root = Path(output_root)
    records = collect_summary_records(
        family=family,
        run_group_id=run_group_id,
        dataset_id=dataset_id,
        output_root=output_root,
    )
    write_local_leaderboard(records, Path(table_root) / "local_summary_leaderboard.csv")
    best = select_best_record(records)
    if best is None:
        return None

    payload = {
        "model_family": family,
        "run_group_id": run_group_id,
        "dataset_id": dataset_id,
        "config_path": str(config_path),
        "dataset_spec_path": str(dataset_spec_path),
        "summary_path": best["summary_path"],
        "artifact_path": best.get("artifact_path"),
        "model_name": best["model_name"],
        "validation_metric_name": best["validation_metric_name"],
        "validation_metric": best["validation_metric"],
        "selected_threshold": best.get("selected_threshold"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    registry_path = selection_registry_path(family, repo_root=repo_root)
    registry = read_selection_registry(registry_path)
    registry[run_group_id] = payload
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def collect_summary_records(
    *,
    family: str,
    run_group_id: str,
    dataset_id: str,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    records = []
    for summary_path in sorted(Path(output_root).glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        score = extract_validation_score(summary)
        if score is None:
            continue
        records.append(
            {
                "family": family,
                "run_group_id": run_group_id,
                "dataset_id": dataset_id,
                "model_name": infer_model_name(summary, summary_path),
                "summary_path": str(summary_path),
                "artifact_path": summary.get("model_path"),
                "validation_metric_name": PRIMARY_PAUC_METRIC,
                "validation_metric": score,
                "selected_threshold": summary.get("selected_threshold"),
                "duration_seconds": summary.get("duration_seconds"),
            }
        )
    return records


def extract_validation_score(summary: dict[str, Any]) -> float | None:
    if "metrics" in summary and isinstance(summary["metrics"], dict):
        val_metrics = summary["metrics"].get("val", {})
        score = val_metrics.get(PRIMARY_PAUC_METRIC)
        if is_valid_number(score):
            return float(score)
    best_validation_metrics = summary.get("best_validation_metrics", {})
    if isinstance(best_validation_metrics, dict):
        score = best_validation_metrics.get(PRIMARY_PAUC_METRIC)
        if is_valid_number(score):
            return float(score)
    score = summary.get("best_validation_metric")
    if is_valid_number(score):
        return float(score)
    return None


def infer_model_name(summary: dict[str, Any], summary_path: Path) -> str:
    if summary.get("model_name"):
        return str(summary["model_name"])
    if summary_path.parent.name == "best_final_test" and summary_path.parent.parent.name:
        return summary_path.parent.parent.name
    return summary_path.parent.name


def select_best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(records, key=lambda record: float(record["validation_metric"]))


def write_local_leaderboard(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "family",
        "run_group_id",
        "dataset_id",
        "model_name",
        "validation_metric_name",
        "validation_metric",
        "selected_threshold",
        "duration_seconds",
        "summary_path",
        "artifact_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["validation_metric"], reverse=True):
            writer.writerow({column: record.get(column, "") for column in columns})


def is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)

