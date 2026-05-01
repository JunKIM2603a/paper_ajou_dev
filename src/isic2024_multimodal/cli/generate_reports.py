from __future__ import annotations

import argparse
import subprocess
import sys

from isic2024_multimodal.utils.runtime_env import get_default_mlflow_tracking_uri

PRIMARY_PAUC_METRIC = "pauc_above_tpr80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLflow CSV and HTML reports for ISIC2024 experiments.")
    parser.add_argument("--tracking-uri", default=get_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Baselines")
    parser.add_argument("--output-prefix", default="experiments/tables/mlflow_report")
    parser.add_argument("--sort-metric", default=f"best_{PRIMARY_PAUC_METRIC}")
    parser.add_argument("--parent-sort-metric", default=f"best_{PRIMARY_PAUC_METRIC}")
    parser.add_argument("--child-sort-metric", default=f"val_{PRIMARY_PAUC_METRIC}")
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--model-family", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_output = f"{args.output_prefix}.csv"
    html_output = f"{args.output_prefix}.html"
    commands = [
        [
            sys.executable,
            "-m",
            "isic2024_multimodal.reporting.mlflow_report",
            "--tracking-uri",
            args.tracking_uri,
            "--experiment-name",
            args.experiment_name,
            "--run-group-id",
            args.run_group_id or "",
            "--dataset-id",
            args.dataset_id or "",
            "--model-family",
            args.model_family or "",
            "--sort-metric",
            args.sort_metric,
            "--output",
            csv_output,
        ],
        [
            sys.executable,
            "-m",
            "isic2024_multimodal.reporting.mlflow_html_report",
            "--tracking-uri",
            args.tracking_uri,
            "--experiment-name",
            args.experiment_name,
            "--run-group-id",
            args.run_group_id or "",
            "--dataset-id",
            args.dataset_id or "",
            "--model-family",
            args.model_family or "",
            "--parent-sort-metric",
            args.parent_sort_metric,
            "--child-sort-metric",
            args.child_sort_metric,
            "--output",
            html_output,
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
