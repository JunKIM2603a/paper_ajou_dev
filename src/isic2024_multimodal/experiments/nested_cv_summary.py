from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isic2024_multimodal.evaluation.metrics import PRIMARY_PAUC_METRIC


DEFAULT_VALIDATION_METRIC_ORDER = [
    PRIMARY_PAUC_METRIC,
    "auc_roc",
    "average_precision",
    "f1_score",
    "balanced_accuracy",
]

DEFAULT_TEST_METRICS = [
    PRIMARY_PAUC_METRIC,
    "auc_roc",
    "average_precision",
    "f1_score",
    "precision",
    "recall",
    "balanced_accuracy",
    "false_negative_count",
    "false_positive_count",
]


@dataclass(frozen=True)
class NestedCVSummaryRecord:
    family: str
    run_group_id: str
    model_name: str
    outer_fold: int
    inner_fold: int
    validation_metric_name: str
    validation_metric: float
    test_metrics: dict[str, float]
    selected_threshold: float | None
    threshold_source: str | None
    summary_path: str
    feature_set: str | None = None
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "family": self.family,
            "run_group_id": self.run_group_id,
            "model_name": self.model_name,
            "feature_set": self.feature_set or "",
            "max_train_rows": "" if self.max_train_rows is None else self.max_train_rows,
            "max_val_rows": "" if self.max_val_rows is None else self.max_val_rows,
            "max_test_rows": "" if self.max_test_rows is None else self.max_test_rows,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "validation_metric_name": self.validation_metric_name,
            "validation_metric": self.validation_metric,
            "selected_threshold": "" if self.selected_threshold is None else self.selected_threshold,
            "threshold_source": self.threshold_source or "",
            "summary_path": self.summary_path,
        }
        for metric_name in DEFAULT_TEST_METRICS:
            row[f"test_{metric_name}"] = self.test_metrics.get(metric_name, "")
        return row


def collect_nested_cv_summary_records(
    *,
    output_root: str | Path,
    family: str,
    run_group_id: str,
    validation_metric_order: list[str] | None = None,
) -> list[NestedCVSummaryRecord]:
    output_root = Path(output_root)
    metric_order = validation_metric_order or DEFAULT_VALIDATION_METRIC_ORDER
    records: list[NestedCVSummaryRecord] = []
    for summary_path in candidate_summary_paths(output_root):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parsed = parse_nested_cv_summary(
            summary=summary,
            summary_path=summary_path,
            family=family,
            run_group_id=run_group_id,
            validation_metric_order=metric_order,
        )
        if parsed is not None:
            records.append(parsed)
    return records


def candidate_summary_paths(output_root: Path) -> list[Path]:
    summary_paths = sorted(output_root.glob("**/summary.json"))
    best_final_paths = [path for path in summary_paths if path.parent.name == "best_final_test"]
    return best_final_paths or summary_paths


def parse_nested_cv_summary(
    *,
    summary: dict[str, Any],
    summary_path: str | Path,
    family: str,
    run_group_id: str,
    validation_metric_order: list[str] | None = None,
) -> NestedCVSummaryRecord | None:
    split_summary = summary.get("split_summary", {}) if isinstance(summary.get("split_summary"), dict) else {}
    outer_fold = first_int(summary.get("outer_fold"), split_summary.get("outer_fold"))
    inner_fold = first_int(summary.get("inner_fold"), split_summary.get("inner_fold"))
    if outer_fold is None or inner_fold is None:
        return None

    val_metrics = validation_metrics(summary)
    test_metrics = outer_test_metrics(summary)
    if not test_metrics:
        return None

    metric_name, metric_value = select_validation_metric(
        val_metrics,
        validation_metric_order or DEFAULT_VALIDATION_METRIC_ORDER,
    )
    if metric_name is None or metric_value is None:
        return None

    hyperparameters = summary.get("hyperparameters", {}) if isinstance(summary.get("hyperparameters"), dict) else {}
    feature_set = hyperparameters.get("feature_set")
    return NestedCVSummaryRecord(
        family=family,
        run_group_id=run_group_id,
        model_name=infer_model_name(summary, Path(summary_path)),
        feature_set=str(feature_set) if feature_set else None,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        validation_metric_name=metric_name,
        validation_metric=metric_value,
        test_metrics=test_metrics,
        selected_threshold=first_float(summary.get("selected_threshold")),
        threshold_source=first_str(summary.get("threshold_source"), split_summary.get("threshold_source")),
        summary_path=str(summary_path),
        max_train_rows=first_int(summary.get("max_train_rows"), split_summary.get("max_train_rows")),
        max_val_rows=first_int(summary.get("max_val_rows"), split_summary.get("max_val_rows")),
        max_test_rows=first_int(summary.get("max_test_rows"), split_summary.get("max_test_rows")),
    )


def validation_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("val"), dict):
        return metrics["val"]
    best_validation_metrics = summary.get("best_validation_metrics")
    if isinstance(best_validation_metrics, dict):
        return best_validation_metrics
    return {}


def outer_test_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics")
    raw_metrics: dict[str, Any] = {}
    if isinstance(metrics, dict) and isinstance(metrics.get("test"), dict):
        raw_metrics = metrics["test"]
    elif isinstance(summary.get("test_metrics"), dict):
        raw_metrics = summary["test_metrics"]
    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in raw_metrics.items()
        if is_valid_number(metric_value)
    }


def select_validation_metric(
    metrics: dict[str, Any],
    metric_order: list[str],
) -> tuple[str | None, float | None]:
    for metric_name in metric_order:
        metric_value = metrics.get(metric_name)
        if is_valid_number(metric_value):
            return metric_name, float(metric_value)
    return None, None


def select_outer_fold_records(
    records: list[NestedCVSummaryRecord],
    *,
    validation_metric_order: list[str] | None = None,
) -> list[NestedCVSummaryRecord]:
    selected_by_outer: dict[int, NestedCVSummaryRecord] = {}
    for record in records:
        current = selected_by_outer.get(record.outer_fold)
        record_key = selection_key(record, validation_metric_order=validation_metric_order)
        current_key = None
        if current is not None:
            current_key = selection_key(current, validation_metric_order=validation_metric_order)
        if current is None or current_key is None or record_key > current_key:
            selected_by_outer[record.outer_fold] = record
    return [selected_by_outer[outer_fold] for outer_fold in sorted(selected_by_outer)]


def selection_key(
    record: NestedCVSummaryRecord,
    *,
    validation_metric_order: list[str] | None = None,
) -> tuple[int, float, int, str]:
    # Validation metric is the only score-bearing selection signal. The remaining
    # fields make ties deterministic without looking at outer-test metrics.
    metric_order = validation_metric_order or DEFAULT_VALIDATION_METRIC_ORDER
    return (
        -validation_metric_rank(record.validation_metric_name, metric_order),
        record.validation_metric,
        -record.inner_fold,
        record.model_name,
    )


def validation_metric_rank(metric_name: str, metric_order: list[str]) -> int:
    try:
        return metric_order.index(metric_name)
    except ValueError:
        return len(metric_order)


def summarize_selected_test_metrics(
    selected_records: list[NestedCVSummaryRecord],
    *,
    metric_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in metric_names or DEFAULT_TEST_METRICS:
        values = [
            record.test_metrics[metric_name]
            for record in selected_records
            if metric_name in record.test_metrics and is_valid_number(record.test_metrics[metric_name])
        ]
        if not values:
            continue
        rows.append(
            {
                "metric": metric_name,
                "fold_count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def write_nested_cv_summary_outputs(
    *,
    records: list[NestedCVSummaryRecord],
    table_root: str | Path,
    family: str,
    run_group_id: str,
    expected_outer_folds: int | None = 5,
    validation_metric_order: list[str] | None = None,
) -> dict[str, Any]:
    table_root = Path(table_root)
    table_root.mkdir(parents=True, exist_ok=True)
    metric_order = validation_metric_order or DEFAULT_VALIDATION_METRIC_ORDER
    selected_records = select_outer_fold_records(records, validation_metric_order=metric_order)
    metric_summary = summarize_selected_test_metrics(selected_records)
    manifest = {
        "family": family,
        "run_group_id": run_group_id,
        "protocol": "outer_fold_validation_selected_test_summary_v1",
        "selection_rule": "select one record per outer_fold by validation metric only",
        "refit_performed_by_summarizer": False,
        "validation_metric_order": metric_order,
        "expected_outer_folds": expected_outer_folds,
        "candidate_record_count": len(records),
        "selected_outer_fold_count": len(selected_records),
        "missing_outer_folds": missing_outer_folds(selected_records, expected_outer_folds),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {
            "candidate_table": str(table_root / "nested_cv_all_candidates.csv"),
            "outer_selection_table": str(table_root / "nested_cv_outer_selection.csv"),
            "metric_summary_table": str(table_root / "nested_cv_metric_summary.csv"),
            "markdown_summary": str(table_root / "nested_cv_summary.md"),
            "json_summary": str(table_root / "nested_cv_summary.json"),
        },
    }
    write_records_csv(records, table_root / "nested_cv_all_candidates.csv")
    write_records_csv(selected_records, table_root / "nested_cv_outer_selection.csv")
    write_metric_summary_csv(metric_summary, table_root / "nested_cv_metric_summary.csv")
    (table_root / "nested_cv_summary.json").write_text(
        json.dumps(
            {
                **manifest,
                "selected_outer_folds": [record.to_row() for record in selected_records],
                "metric_summary": metric_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (table_root / "nested_cv_summary.md").write_text(
        render_markdown_summary(
            manifest=manifest,
            selected_records=selected_records,
            metric_summary=metric_summary,
        ),
        encoding="utf-8",
    )
    return manifest


def write_records_csv(records: list[NestedCVSummaryRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "family",
        "run_group_id",
        "model_name",
        "feature_set",
        "max_train_rows",
        "max_val_rows",
        "max_test_rows",
        "outer_fold",
        "inner_fold",
        "validation_metric_name",
        "validation_metric",
        "selected_threshold",
        "threshold_source",
        *[f"test_{metric_name}" for metric_name in DEFAULT_TEST_METRICS],
        "summary_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.outer_fold, item.inner_fold, item.model_name)):
            writer.writerow({column: record.to_row().get(column, "") for column in columns})


def write_metric_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["metric", "fold_count", "mean", "std", "min", "max"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def render_markdown_summary(
    *,
    manifest: dict[str, Any],
    selected_records: list[NestedCVSummaryRecord],
    metric_summary: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Nested CV Summary: {manifest['run_group_id']}",
        "",
        f"- Family: `{manifest['family']}`",
        f"- Protocol: `{manifest['protocol']}`",
        f"- Candidate records: {manifest['candidate_record_count']}",
        f"- Selected outer folds: {manifest['selected_outer_fold_count']}",
        "- Refit performed by summarizer: false",
    ]
    if manifest["missing_outer_folds"]:
        lines.append(f"- Missing outer folds: {manifest['missing_outer_folds']}")
    if has_row_caps(selected_records):
        lines.append("- Row caps detected: this is a capped/smoke summary, not a full-data paper result.")
    lines.extend(
        [
            "",
            "## Outer Fold Selection",
            "",
            "| outer_fold | inner_fold | model | feature_set | validation_metric | validation_value | threshold_source |",
            "|---:|---:|---|---|---|---:|---|",
        ]
    )
    for record in selected_records:
        lines.append(
            "| "
            f"{record.outer_fold} | {record.inner_fold} | {record.model_name} | "
            f"{record.feature_set or ''} | {record.validation_metric_name} | "
            f"{record.validation_metric:.6g} | {record.threshold_source or ''} |"
        )
    lines.extend(
        [
            "",
            "## Selected Outer-Test Metrics",
            "",
            "| metric | folds | mean | std | min | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_summary:
        lines.append(
            "| "
            f"{row['metric']} | {row['fold_count']} | {row['mean']:.6g} | "
            f"{row['std']:.6g} | {row['min']:.6g} | {row['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Selection uses validation metrics only. Outer-test metrics are summarized after the validation-based selection.",
            "This summary tool does not refit models on the full outer-train partition.",
            "",
        ]
    )
    return "\n".join(lines)


def has_row_caps(records: list[NestedCVSummaryRecord]) -> bool:
    return any(
        record.max_train_rows is not None
        or record.max_val_rows is not None
        or record.max_test_rows is not None
        for record in records
    )


def missing_outer_folds(records: list[NestedCVSummaryRecord], expected_outer_folds: int | None) -> list[int]:
    if expected_outer_folds is None:
        return []
    present = {record.outer_fold for record in records}
    return [outer_fold for outer_fold in range(expected_outer_folds) if outer_fold not in present]


def infer_model_name(summary: dict[str, Any], summary_path: Path) -> str:
    if summary.get("model_name"):
        return str(summary["model_name"])
    if summary_path.parent.name == "best_final_test" and summary_path.parent.parent.name:
        return summary_path.parent.parent.name
    return summary_path.parent.name


def first_int(*values: Any) -> int | None:
    for value in values:
        if value in ("", None):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def first_float(*values: Any) -> float | None:
    for value in values:
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            return number
    return None


def first_str(*values: Any) -> str | None:
    for value in values:
        if value not in ("", None):
            return str(value)
    return None


def is_valid_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)
