from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLflow best-run leaderboard.")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="CBIS-DDSM-Benchmark")
    parser.add_argument("--output", default="artifacts/mlflow_leaderboard.csv")
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
        order_by=["metrics.best_auc_roc DESC"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "tags.model_name",
        "metrics.best_accuracy",
        "metrics.best_precision",
        "metrics.best_recall",
        "metrics.best_f1_score",
        "metrics.best_auc_roc",
        "params.best_hp_learning_rate",
        "params.best_hp_weight_decay",
        "params.best_hp_epochs",
        "tags.best_child_run_name",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model_name", "accuracy", "precision", "recall", "f1_score",
            "auc_roc", "learning_rate", "weight_decay", "epochs", "best_child_run_name",
        ])
        for _, row in runs.iterrows():
            writer.writerow([row.get(column, "") for column in columns])

    print(f"Saved leaderboard to {output_path}")


if __name__ == "__main__":
    main()
