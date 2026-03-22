from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cbis_ddsm_benchmark.config_utils import expand_search_space, load_json, sanitize_run_name
from cbis_ddsm_benchmark.data import CbisDdsmDataset, build_manifest, create_splits, resolve_dataset_root
from cbis_ddsm_benchmark.models import build_model
from cbis_ddsm_benchmark.reproducibility import (
    DEFAULT_SEED,
    make_torch_generator,
    make_worker_init_fn,
    set_global_seed,
)
from cbis_ddsm_benchmark.trainer import run_training


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
        default=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
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
        "--seed",
        type=int,
        default=None,
        help="Override the seed in config.json. Defaults to the config seed or 42.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    resolved_dataset_root = resolve_dataset_root(args.dataset_root)
    _log(f"config={args.config}")
    _log(f"requested dataset_root={args.dataset_root}")
    _log(f"resolved dataset_root={resolved_dataset_root}")
    _log(f"device={args.device}, output_root={args.output_root}")
    # 사전학습 가중치와 허깅페이스 캐시가 사용자 홈이 아닌 프로젝트 내부에 쌓이도록 고정한다.
    cache_root = Path(args.output_root) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_root / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf"))
    config = load_json(args.config)
    model_name = config["model"]["display_name"]
    image_size = int(config["dataset"].get("image_size", 224))
    batch_size = int(config["dataset"].get("batch_size", 16))
    num_workers = int(config["dataset"].get("num_workers", 0))
    validation_ratio = float(config["dataset"].get("validation_ratio", 0.2))
    seed = int(args.seed if args.seed is not None else config["dataset"].get("seed", DEFAULT_SEED))
    config.setdefault("dataset", {})
    config["dataset"]["seed"] = seed
    set_global_seed(seed)
    _log(
        f"loaded config for model={model_name}, image_size={image_size}, batch_size={batch_size}, "
        f"num_workers={num_workers}, validation_ratio={validation_ratio}, seed={seed}"
    )

    # CSV와 JPEG 매핑 정보를 한 번 정리한 뒤 캐시해 두면 반복 실행 시 로딩이 빨라진다.
    _log("building manifest")
    manifest = build_manifest(
        resolved_dataset_root,
        cache_path=Path(args.output_root) / "cache" / "cbis_ddsm_manifest.json",
    )
    _log(f"manifest ready: {len(manifest)} samples")
    splits = create_splits(manifest, validation_ratio=validation_ratio, seed=seed)
    _log(
        f"splits ready: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    train_dataset = CbisDdsmDataset(splits["train"], image_size=image_size, augment=True)
    val_dataset = CbisDdsmDataset(splits["val"], image_size=image_size, augment=False)
    test_dataset = CbisDdsmDataset(splits["test"], image_size=image_size, augment=False)

    # train/val/test 모두 같은 샘플 정의를 쓰되, shuffle 여부만 다르게 둔다.
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
            generator=make_torch_generator(seed),
            worker_init_fn=make_worker_init_fn(seed),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
            generator=make_torch_generator(seed + 1),
            worker_init_fn=make_worker_init_fn(seed + 1000),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
            generator=make_torch_generator(seed + 2),
            worker_init_fn=make_worker_init_fn(seed + 2000),
        ),
    }
    _log("dataloaders created")

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
    _log(f"trial combinations={len(combinations)}")
    best_result = None
    best_run_name = None

    # 부모 런은 모델 대표 결과, 자식 런은 하이퍼파라미터 trial 기록용이다.
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
                "resolved_dataset_root": str(resolved_dataset_root.resolve()),
                "image_size": image_size,
                "batch_size": batch_size,
                "validation_ratio": validation_ratio,
                "seed": seed,
                "num_train_samples": len(train_dataset),
                "num_val_samples": len(val_dataset),
                "num_test_samples": len(test_dataset),
            }
        )
        mlflow.log_dict(config, "resolved_config.json")

        # search_space의 모든 조합을 순회하면서 자식 런을 생성한다.
        for index, hyperparameters in enumerate(combinations, start=1):
            run_name = f"{parent_run_name}_trial_{index:03d}"
            output_dir = Path(args.output_root) / parent_run_name / run_name
            trial_seed = seed + index - 1
            _log(f"trial {index}/{len(combinations)} start: {run_name}")
            _log(f"trial seed={trial_seed}")
            _log(f"trial hyperparameters={json.dumps(hyperparameters, ensure_ascii=False)}")
            _log("building model")
            set_global_seed(trial_seed)
            model = build_model(
                {
                    **config["model"],
                    "image_size": image_size,
                },
                num_classes=2,
            )
            _log("model ready")

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
                mlflow.log_param("trial_seed", trial_seed)

                # 각 trial의 산출물은 artifacts/<모델명>/<trial명> 아래에 저장된다.
                summary = run_training(
                    model=model,
                    dataloaders=dataloaders,
                    device=args.device,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    mlflow_client=mlflow,
                )
                _log(f"trial complete: {run_name}")

                for metric_name, metric_value in summary["test_metrics"].items():
                    mlflow.log_metric(f"test_{metric_name}", float(metric_value))
                mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
                mlflow.log_metric("best_epoch", float(summary["best_epoch"]))
                mlflow.log_artifact(str(output_dir / "history.csv"))
                mlflow.log_artifact(str(output_dir / "summary.json"))
                mlflow.log_artifact(str(output_dir / "best_model.pt"))

                score = summary["test_metrics"]["auc_roc"]
                if score != score:
                    score = summary["test_metrics"]["f1_score"]
                # 부모 런에는 자식 런 중 최고 성능 조합만 요약해 남긴다.
                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "summary": summary,
                        "hyperparameters": hyperparameters,
                        "trial_seed": trial_seed,
                    }
                    best_run_name = run_name

        if best_result is None:
            raise RuntimeError("No successful runs were completed.")

        mlflow.log_metrics(
            {
                f"best_{name}": float(value)
                for name, value in best_result["summary"]["test_metrics"].items()
            }
        )
        mlflow.log_params({f"best_hp_{key}": value for key, value in best_result["hyperparameters"].items()})
        mlflow.log_param("best_trial_seed", best_result["trial_seed"])
        mlflow.set_tag("best_child_run_name", best_run_name)
        mlflow.log_dict(
            {
                "best_child_run_name": best_run_name,
                "best_child_seed": best_result["trial_seed"],
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


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[run_experiment {timestamp}] {message}", flush=True)


if __name__ == "__main__":
    main()









