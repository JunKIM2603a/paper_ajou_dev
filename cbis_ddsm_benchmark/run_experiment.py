from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cbis_ddsm_benchmark.config_utils import expand_search_space, load_json, sanitize_run_name
from cbis_ddsm_benchmark.data import CbisDdsmDataset, build_manifest, create_splits
from cbis_ddsm_benchmark.models import build_model
from cbis_ddsm_benchmark.trainer import run_training


TRANSFER_STRATEGIES = ("full_finetune", "linear_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CBIS-DDSM model benchmark.")
    parser.add_argument("--config", required=True, help="Path to the model config JSON file.")
    parser.add_argument(
        "--dataset-root",
        default="dataset/archive_CBIS-DDSM_kaggle",
        help="Path to the extracted CBIS-DDSM dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts",
        help="Directory for checkpoints, histories, and summaries.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="CBIS-DDSM-Benchmark",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--transfer-strategy",
        choices=TRANSFER_STRATEGIES,
        default=None,
        help="Override config model.transfer_strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_root = Path(args.output_root) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_root / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf"))

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    config = _resolve_config_paths(config, config_path.parent)
    if args.transfer_strategy is not None:
        config.setdefault("model", {})["transfer_strategy"] = args.transfer_strategy

    model_name = config["model"]["display_name"]
    image_size = int(config["dataset"].get("image_size", 224))
    batch_size = int(config["dataset"].get("batch_size", 16))
    num_workers = int(config["dataset"].get("num_workers", 0))
    validation_ratio = float(config["dataset"].get("validation_ratio", 0.2))
    seed = int(config["dataset"].get("seed", 42))

    manifest = build_manifest(
        args.dataset_root,
        cache_path=Path(args.output_root) / "cache" / "cbis_ddsm_manifest.json",
    )
    splits = create_splits(manifest, validation_ratio=validation_ratio, seed=seed)

    train_dataset = CbisDdsmDataset(splits["train"], image_size=image_size, augment=True)
    val_dataset = CbisDdsmDataset(splits["val"], image_size=image_size, augment=False)
    test_dataset = CbisDdsmDataset(splits["test"], image_size=image_size, augment=False)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
    }

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required to run this benchmark. Install dependencies from requirements.txt first."
        ) from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    parent_run_name = sanitize_run_name(model_name)
    search_space = config.get("search_space", {})
    combinations = expand_search_space(search_space)
    best_result: dict[str, Any] | None = None
    best_run_name: str | None = None

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags(
            {
                "model_name": model_name,
                "benchmark": "CBIS-DDSM",
                "role": "model_parent",
            }
        )
        mlflow.log_params(
            {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "config_path": str(config_path),
                "model_backend": config["model"]["backend"],
                **flatten_params(config["model"], prefix="model"),
                **flatten_params(config["dataset"], prefix="dataset"),
            }
        )
        mlflow.log_dict(config, "resolved_config.json")

        for index, hyperparameters in enumerate(combinations, start=1):
            run_name = f"{parent_run_name}_trial_{index:03d}"
            output_dir = Path(args.output_root) / parent_run_name / run_name
            model = build_model(config["model"], num_classes=2)

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tags(
                    {
                        "model_name": model_name,
                        "benchmark": "CBIS-DDSM",
                        "role": "hyperparameter_trial",
                    }
                )
                mlflow.log_params(flatten_params(config["model"], prefix="model"))
                mlflow.log_params({f"hp_{key}": value for key, value in hyperparameters.items()})

                summary = run_training(
                    model=model,
                    dataloaders=dataloaders,
                    device=args.device,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    mlflow_client=mlflow,
                )

                for metric_name, metric_value in summary["test_metrics"].items():
                    mlflow.log_metric(f"test_{metric_name}", float(metric_value))
                mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
                mlflow.log_metric("best_epoch", float(summary["best_epoch"]))
                mlflow.log_metric("trainable_parameter_count", float(summary["trainable_parameter_count"]))
                mlflow.log_metric("frozen_parameter_count", float(summary["frozen_parameter_count"]))
                mlflow.log_artifact(str(output_dir / "history.csv"))
                mlflow.log_artifact(str(output_dir / "summary.json"))
                mlflow.log_artifact(str(output_dir / "best_model.pt"))

                score = summary["test_metrics"]["auc_roc"]
                if score != score:
                    score = summary["test_metrics"]["f1_score"]

                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "summary": summary,
                        "hyperparameters": hyperparameters,
                    }
                    best_run_name = run_name

        if best_result is None or best_run_name is None:
            raise RuntimeError("No successful runs were completed.")

        mlflow.log_metrics({f"best_{name}": float(value) for name, value in best_result["summary"]["test_metrics"].items()})
        mlflow.log_params({f"best_hp_{key}": value for key, value in best_result["hyperparameters"].items()})
        mlflow.set_tag("best_child_run_name", best_run_name)
        mlflow.log_dict(
            {
                "best_child_run_name": best_run_name,
                "best_hyperparameters": best_result["hyperparameters"],
                "best_summary": best_result["summary"],
            },
            "best_result.json",
        )

    print(json.dumps({"parent_run_id": parent_run.info.run_id, "best_child_run_name": best_run_name}, indent=2))


def flatten_params(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}_{key}"
        if isinstance(value, dict):
            flattened.update(flatten_params(value, prefix=name))
        else:
            flattened[name] = value
    return flattened


def _resolve_config_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    checkpoint_path = resolved.get("model", {}).get("checkpoint_path")
    if checkpoint_path:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(checkpoint_path))))
        if not expanded.is_absolute():
            expanded = (config_dir / expanded).resolve()
        resolved["model"]["checkpoint_path"] = str(expanded)
    return resolved


if __name__ == "__main__":
    main()
