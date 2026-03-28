from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLflow best-run leaderboard.")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Benchmark")
    parser.add_argument("--output", default="artifacts/mlflow_leaderboard.csv")
    parser.add_argument("--sort-metric", default="best_average_precision")
    return parser.parse_args()


def main() -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required to generate the leaderboard.") from exc

    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {args.experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.role = 'model_parent'",
        order_by=[f"metrics.{args.sort_metric} DESC", "attributes.start_time DESC"],
    )
    rows = _select_best_parent_rows(runs, args.sort_metric)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "tags.model_name",
        "params.best_feature_set",
        f"metrics.{args.sort_metric}",
        "metrics.best_average_precision",
        "metrics.best_balanced_accuracy",
        "metrics.best_accuracy",
        "metrics.best_precision",
        "metrics.best_recall",
        "metrics.best_f1_score",
        "metrics.best_auc_roc",
        "tags.best_child_run_name",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model_name",
                "feature_set",
                args.sort_metric,
                "average_precision",
                "balanced_accuracy",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "best_child_run_name",
            ]
        )
        for row in rows:
            model_name = row.get("tags.model_name", "")
            primary_metric = row.get(f"metrics.{args.sort_metric}", "")
            if _is_missing_value(model_name) or _is_missing_value(primary_metric):
                continue
            writer.writerow([row.get(column, "") for column in columns])

    print(f"Saved leaderboard to {output_path}")


def _is_missing_value(value: object) -> bool:
    if value in ("", None):
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _select_best_parent_rows(runs, sort_metric: str) -> list:
    selected = []
    seen_models: set[str] = set()
    metric_key = f"metrics.{sort_metric}"
    for _, row in runs.iterrows():
        model_name = row.get("tags.model_name", "")
        if _is_missing_value(model_name) or _is_missing_value(row.get(metric_key, "")):
            continue
        model_name = str(model_name)
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        selected.append(row)
    return selected


if __name__ == "__main__":
    main()
