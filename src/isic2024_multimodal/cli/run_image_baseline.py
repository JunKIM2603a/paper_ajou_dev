from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isic2024_multimodal.utils.config_utils import expand_search_space, load_json, sanitize_run_name
from isic2024_multimodal.evaluation.metrics import PRIMARY_PAUC_METRIC
from isic2024_multimodal.training.reproducibility import (
    DEFAULT_SEED,
    make_torch_generator,
    make_worker_init_fn,
    set_global_seed,
)
from isic2024_multimodal.utils.device import resolve_device
from isic2024_multimodal.utils.runtime_env import ensure_expected_conda_env, get_default_mlflow_tracking_uri, load_project_env

DEFAULT_NESTED_SPLIT_CSV = "data/splits/isic2024_official_train_nested_5x4_seed42.csv"
DEFAULT_HOLDOUT_SPLIT_CSV = "data/splits/isic2024_train_validation_test_split_seed42.csv"
DEFAULT_CV_SPLIT_CSV = "data/splits/isic2024_train_validation_5fold_seed42.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ISIC2024 image baseline experiment.")
    parser.add_argument("--config", required=True, help="Path to the model config JSON file.")
    parser.add_argument(
        "--dataset-root",
        default="data/raw/isic_2024_challenge",
        help="Path to the ISIC2024 challenge dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/outputs/image_baselines",
        help="Directory for checkpoints, histories, and summaries.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=get_default_mlflow_tracking_uri(),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="ISIC2024-Image-Baselines",
        help="MLflow experiment name.",
    )
    parser.add_argument("--run-group-id", default=None, help="Optional run group tag used to scope MLflow reports.")
    parser.add_argument("--dataset-id", default=None, help="Versioned dataset id for registry/report filtering.")
    parser.add_argument("--dataset-spec", default=None, help="Dataset spec JSON path used for this run.")
    parser.add_argument("--model-family", default="image_baselines", help="Experiment family tag.")
    parser.add_argument("--split-protocol", choices=["nested_cv", "legacy_holdout"], default="nested_cv")
    parser.add_argument("--nested-split-csv", default=DEFAULT_NESTED_SPLIT_CSV)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--holdout-split-csv", default=DEFAULT_HOLDOUT_SPLIT_CSV)
    parser.add_argument("--cv-split-csv", default=DEFAULT_CV_SPLIT_CSV)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use `auto`, `cpu`, or `cuda`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in config.json. Defaults to the config seed or 42.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit the number of hyperparameter trials for smoke testing.",
    )
    parser.add_argument(
        "--trial-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional zero-based hyperparameter trial indices to run, e.g. --trial-indices 0 1.",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Override the epoch count in search_space for quick validation runs.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the number of training samples after the split is created.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap the number of validation samples after the split is created.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Cap the number of test samples after the split is created.",
    )
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Disable pretrained/model hub weights for smoke testing or offline runs.",
    )
    return parser.parse_args()


def main() -> None:
    load_project_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    ensure_expected_conda_env()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from isic2024_multimodal.data.image_dataset import (
        ImageClassificationDataset,
        build_manifest,
        create_splits_from_nested_csv,
        create_splits_from_locked_csvs,
        resolve_dataset_root,
    )
    from isic2024_multimodal.models.image.checkpoint_preflight import preflight_image_model_config
    from isic2024_multimodal.models.image.factory import build_model
    from isic2024_multimodal.training.trainer import run_training

    device_resolution = resolve_device(args.device)
    args.requested_device = device_resolution.requested_device
    args.device = device_resolution.resolved_device
    args.device_fallback_reason = device_resolution.fallback_reason
    args.cuda_available = device_resolution.cuda_available
    args.visible_device_count = device_resolution.visible_device_count
    resolved_dataset_root = resolve_dataset_root(args.dataset_root)
    _log(f"config={args.config}")
    _log(f"requested dataset_root={args.dataset_root}")
    _log(f"resolved dataset_root={resolved_dataset_root}")
    _log(
        f"requested_device={args.requested_device}, resolved_device={args.device}, "
        f"fallback={args.device_fallback_reason or 'none'}, output_root={args.output_root}"
    )
    _validate_runtime_device(args.device, requested_device=args.requested_device)
    # 사전학습 가중치와 허깅페이스 캐시가 사용자 홈이 아닌 프로젝트 내부에 쌓이도록 고정한다.
    cache_root = Path(args.output_root) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_root / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf"))
    config = load_json(args.config)
    if args.disable_pretrained:
        _disable_pretrained_weights(config)
    checkpoint_preflight = preflight_image_model_config(config["model"])
    _log(f"checkpoint preflight={json.dumps(checkpoint_preflight, ensure_ascii=False)}")
    if args.epochs_override is not None:
        config.setdefault("search_space", {})
        config["search_space"]["epochs"] = [int(args.epochs_override)]
    model_name = config["model"]["display_name"]
    image_size = int(config["dataset"].get("image_size", 224))
    batch_size = int(config["dataset"].get("batch_size", 16))
    num_workers = int(config["dataset"].get("num_workers", 0))
    validation_ratio = float(config["dataset"].get("validation_ratio", 0.2))
    test_ratio = float(config["dataset"].get("test_ratio", 0.2))
    preprocessing_contract = image_preprocessing_contract(config["model"], config["dataset"])
    normalize_mean = preprocessing_contract["normalize_mean"]
    normalize_std = preprocessing_contract["normalize_std"]
    seed = int(args.seed if args.seed is not None else config["dataset"].get("seed", DEFAULT_SEED))
    config.setdefault("dataset", {})
    config["dataset"]["seed"] = seed
    set_global_seed(seed)
    _log(
        f"loaded config for model={model_name}, image_size={image_size}, batch_size={batch_size}, "
        f"num_workers={num_workers}, validation_ratio={validation_ratio}, test_ratio={test_ratio}, seed={seed}"
    )
    _log(f"image preprocessing contract={json.dumps(preprocessing_contract, ensure_ascii=False)}")

    # CSV와 JPEG 매핑 정보를 한 번 정리한 뒤 캐시해 두면 반복 실행 시 로딩이 빨라진다.
    _log("building manifest")
    manifest = build_manifest(
        resolved_dataset_root,
        cache_path=Path(args.output_root) / "cache" / "isic2024_challenge_image_manifest.json",
    )
    _log(f"manifest ready: {len(manifest)} samples")
    if args.split_protocol == "legacy_holdout":
        splits = create_splits_from_locked_csvs(
            manifest,
            holdout_split_csv=args.holdout_split_csv,
            cv_split_csv=args.cv_split_csv,
            cv_fold=args.cv_fold,
        )
        split_source = "locked_split_csv"
    else:
        splits = create_splits_from_nested_csv(
            manifest,
            nested_split_csv=args.nested_split_csv,
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
        )
        split_source = "nested_cv_split_csv"
    split_protocol_audit = image_split_protocol_audit(
        splits,
        nested_protocol=args.split_protocol == "nested_cv",
    )
    splits["train"] = _limit_samples(splits["train"], args.max_train_samples, seed=seed)
    splits["val"] = _limit_samples(splits["val"], args.max_val_samples, seed=seed + 1)
    splits["test"] = _limit_samples(splits["test"], args.max_test_samples, seed=seed + 2)
    _log(
        f"splits ready: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "resolved_dataset_root": str(resolved_dataset_root.resolve()),
                    "dataset_id": args.dataset_id,
                    "dataset_spec_path": args.dataset_spec,
                    "model_family": args.model_family,
                    "requested_device": args.requested_device,
                    "resolved_device": args.device,
                    "effective_device": args.device,
                    "cuda_available": args.cuda_available,
                    "visible_device_count": args.visible_device_count,
                    "device_fallback_reason": args.device_fallback_reason,
                    "run_group_id": args.run_group_id,
                    "split_protocol": args.split_protocol,
                    "split_source": split_source,
                    "nested_split_csv": str(Path(args.nested_split_csv).resolve()),
                    "outer_fold": args.outer_fold,
                    "inner_fold": args.inner_fold,
                    "holdout_split_csv": str(Path(args.holdout_split_csv).resolve()),
                    "cv_split_csv": str(Path(args.cv_split_csv).resolve()),
                    "cv_fold": args.cv_fold,
                    "split_rows": {name: len(items) for name, items in splits.items()},
                    "patient_overlap_audit": split_protocol_audit["patient_overlap_audit"],
                    "triple_balance_audit": split_protocol_audit["triple_balance_audit"],
                    "checkpoint_preflight": checkpoint_preflight,
                    "image_preprocessing_contract": preprocessing_contract,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    train_dataset = ImageClassificationDataset(
        splits["train"],
        image_size=image_size,
        augment=True,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    val_dataset = ImageClassificationDataset(
        splits["val"],
        image_size=image_size,
        augment=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    test_dataset = ImageClassificationDataset(
        splits["test"],
        image_size=image_size,
        augment=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 2

    sampler_strategy = str(config["dataset"].get("sampler_strategy", "patient_balanced_weighted"))
    train_sampler = None
    sampler_report = {"strategy": sampler_strategy, "status": "not_used"}
    if sampler_strategy == "patient_balanced_weighted":
        sampler_weights = patient_balanced_sample_weights(splits["train"])
        train_sampler = WeightedRandomSampler(
            sampler_weights,
            num_samples=len(sampler_weights),
            replacement=True,
            generator=make_torch_generator(seed),
        )
        sampler_report = sampler_weight_report(splits["train"], sampler_weights, strategy=sampler_strategy)
    elif sampler_strategy not in {"none", "shuffle"}:
        raise ValueError(f"Unsupported image train sampler_strategy: {sampler_strategy}")
    _log(f"train sampler={json.dumps(sampler_report, ensure_ascii=False)}")

    # train/val/test 모두 같은 샘플 정의를 쓰되, train만 train-only sampler 또는 shuffle을 사용한다.
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=make_torch_generator(seed),
            worker_init_fn=make_worker_init_fn(seed),
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=make_torch_generator(seed + 1),
            worker_init_fn=make_worker_init_fn(seed + 1000),
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=make_torch_generator(seed + 2),
            worker_init_fn=make_worker_init_fn(seed + 2000),
            **common_loader_kwargs,
        ),
    }
    _log("dataloaders created")

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required to run this experiment. Install dependencies from requirements.txt first."
        ) from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    parent_run_name = sanitize_run_name(model_name)
    search_space = config.get("search_space", {})
    combinations = expand_search_space(search_space)
    planned_trials = _select_trial_plan(
        combinations=combinations,
        trial_indices=args.trial_indices,
        max_trials=args.max_trials,
    )
    _log(f"trial combinations={len(combinations)}, planned_trials={len(planned_trials)}")
    best_result = None
    best_run_name = None

    # 부모 런은 모델 대표 결과, 자식 런은 하이퍼파라미터 trial 기록용이다.
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags(
            {
                "model_name": model_name,
                "experiment_family": "isic2024_image_baseline",
                "role": "model_parent",
                "run_group_id": args.run_group_id or "",
                "dataset_id": args.dataset_id or "",
                "model_family": args.model_family,
            }
        )
        mlflow.log_params(
            {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "resolved_dataset_root": str(resolved_dataset_root.resolve()),
                "image_size": image_size,
                "batch_size": batch_size,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "seed": seed,
                "disable_pretrained": args.disable_pretrained,
                "max_trials": args.max_trials,
                "trial_indices": json.dumps(args.trial_indices) if args.trial_indices is not None else None,
                "epochs_override": args.epochs_override,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "max_test_samples": args.max_test_samples,
                "num_train_samples": len(train_dataset),
                "num_val_samples": len(val_dataset),
                "num_test_samples": len(test_dataset),
                "primary_metric_name": PRIMARY_PAUC_METRIC,
                "run_group_id": args.run_group_id,
                "dataset_id": args.dataset_id,
                "dataset_spec_path": args.dataset_spec,
                "model_family": args.model_family,
                "split_protocol": args.split_protocol,
                "split_source": split_source,
                "nested_split_csv": str(Path(args.nested_split_csv).resolve()),
                "outer_fold": args.outer_fold,
                "inner_fold": args.inner_fold,
                "holdout_split_csv": str(Path(args.holdout_split_csv).resolve()),
                "cv_split_csv": str(Path(args.cv_split_csv).resolve()),
                "cv_fold": args.cv_fold,
                "image_normalize_mean": json.dumps(normalize_mean),
                "image_normalize_std": json.dumps(normalize_std),
                "image_preprocessing_source": preprocessing_contract["source"],
                "image_preprocessing_notes": preprocessing_contract["notes"],
                "train_sampler_strategy": sampler_strategy,
                "requested_device": args.requested_device,
                "resolved_device": args.device,
                "effective_device": args.device,
                "device_fallback_reason": args.device_fallback_reason,
                "cuda_available": args.cuda_available,
                "visible_device_count": args.visible_device_count,
            }
        )
        mlflow.log_dict(checkpoint_preflight, "checkpoint_preflight.json")
        mlflow.log_dict(preprocessing_contract, "image_preprocessing_contract.json")
        mlflow.log_dict(sampler_report, "train_sampler_report.json")
        mlflow.log_dict(config, "resolved_config.json")

        # search_space의 모든 조합을 순회하면서 자식 런을 생성한다.
        for execution_order, (trial_index, hyperparameters) in enumerate(planned_trials, start=1):
            run_name = f"{parent_run_name}_trial_{trial_index + 1:03d}"
            output_dir = Path(args.output_root) / parent_run_name / run_name
            trial_seed = seed + trial_index
            trial_started_at = current_timestamp()
            _log(
                f"trial {execution_order}/{len(planned_trials)} start: {run_name} "
                f"(search_space_index={trial_index})"
            )
            _log(f"trial seed={trial_seed}")
            _log(f"trial hyperparameters={json.dumps(hyperparameters, ensure_ascii=False)}")
            write_run_status(
                output_dir,
                {
                    "status": "running",
                    "model_name": model_name,
                    "config_path": str(Path(args.config).resolve()),
                    "run_name": run_name,
                    "trial_index": trial_index,
                    "trial_seed": trial_seed,
                    "started_at": trial_started_at,
                    "ended_at": None,
                    "duration_seconds": None,
                    "split_rows": {name: len(items) for name, items in splits.items()},
                    "checkpoint_preflight": checkpoint_preflight,
                    "requested_device": args.requested_device,
                    "resolved_device": args.device,
                    "effective_device": args.device,
                    "device_fallback_reason": args.device_fallback_reason,
                    "hyperparameters": hyperparameters,
                },
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tags(
                    {
                        "model_name": model_name,
                        "experiment_family": "isic2024_image_baseline",
                        "role": "hyperparameter_trial",
                        "run_group_id": args.run_group_id or "",
                        "dataset_id": args.dataset_id or "",
                        "model_family": args.model_family,
                    }
                )
                mlflow.log_params(flatten_params(config["model"], prefix="model"))
                mlflow.log_params({f"hp_{key}": value for key, value in hyperparameters.items()})
                mlflow.log_param("trial_seed", trial_seed)
                mlflow.log_param("trial_index", trial_index)
                mlflow.log_param("run_group_id", args.run_group_id)
                mlflow.log_param("dataset_id", args.dataset_id)
                mlflow.log_param("dataset_spec_path", args.dataset_spec)
                mlflow.log_param("model_family", args.model_family)
                mlflow.log_param("requested_device", args.requested_device)
                mlflow.log_param("resolved_device", args.device)
                mlflow.log_param("effective_device", args.device)
                mlflow.log_param("device_fallback_reason", args.device_fallback_reason)

                model = None
                try:
                    _log("building model")
                    set_global_seed(trial_seed)
                    model = build_model(
                        {
                            **config["model"],
                            "image_size": image_size,
                        },
                        num_classes=1,
                    )
                    _log("model ready")

                    # 각 trial의 산출물은 experiments/outputs/image_baselines/<모델명>/<trial명> 아래에 저장된다.
                    summary = run_training(
                        model=model,
                        dataloaders=dataloaders,
                        device=args.device,
                        hyperparameters=hyperparameters,
                        output_dir=output_dir,
                        mlflow_client=mlflow,
                    )
                    summary.update(
                        {
                            "model_name": model_name,
                            "run_group_id": args.run_group_id,
                            "dataset_id": args.dataset_id,
                            "dataset_spec_path": args.dataset_spec,
                            "model_family": args.model_family,
                            "split_protocol": args.split_protocol,
                            "split_source": split_source,
                            "threshold_source": "inner_validation_f1"
                            if args.split_protocol == "nested_cv"
                            else summary.get("threshold_source", "validation_f1"),
                            "nested_split_csv": str(Path(args.nested_split_csv).resolve()),
                            "outer_fold": args.outer_fold,
                            "inner_fold": args.inner_fold,
                            "holdout_split_csv": str(Path(args.holdout_split_csv).resolve()),
                            "cv_split_csv": str(Path(args.cv_split_csv).resolve()),
                            "cv_fold": args.cv_fold,
                            "patient_overlap_audit": split_protocol_audit["patient_overlap_audit"],
                            "triple_balance_audit": split_protocol_audit["triple_balance_audit"],
                            "train_sampler_report": sampler_report,
                            "requested_device": args.requested_device,
                            "resolved_device": args.device,
                            "effective_device": args.device,
                            "device_fallback_reason": args.device_fallback_reason,
                        }
                    )
                    (output_dir / "summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    write_run_status(
                        output_dir,
                        {
                            "status": "completed",
                            "model_name": model_name,
                            "config_path": str(Path(args.config).resolve()),
                            "run_name": run_name,
                            "trial_index": trial_index,
                            "trial_seed": trial_seed,
                            "started_at": trial_started_at,
                            "ended_at": current_timestamp(),
                            "duration_seconds": summary.get("duration_seconds"),
                            "split_rows": {name: len(items) for name, items in splits.items()},
                            "checkpoint_preflight": checkpoint_preflight,
                            "requested_device": args.requested_device,
                            "resolved_device": args.device,
                            "effective_device": args.device,
                            "device_fallback_reason": args.device_fallback_reason,
                            "hyperparameters": hyperparameters,
                            "summary_path": str(output_dir / "summary.json"),
                        },
                    )
                    _log(f"trial complete: {run_name}")

                    for metric_name, metric_value in summary["test_metrics"].items():
                        mlflow.log_metric(f"test_{metric_name}", float(metric_value))
                    for metric_name, metric_value in summary["best_validation_metrics"].items():
                        mlflow.log_metric(f"best_val_{metric_name}", float(metric_value))
                    mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
                    mlflow.log_metric("best_epoch", float(summary["best_epoch"]))
                    mlflow.log_artifact(str(output_dir / "history.csv"))
                    mlflow.log_artifact(str(output_dir / "summary.json"))
                    mlflow.log_artifact(str(output_dir / "best_model.pt"))
                    mlflow.set_tag("trial_status", "completed")

                    score = select_trial_score(summary)
                    # 부모 런에는 자식 런 중 최고 성능 조합만 요약해 남긴다.
                    if best_result is None or score > best_result["score"]:
                        best_result = {
                            "score": score,
                            "summary": summary,
                            "hyperparameters": hyperparameters,
                            "trial_seed": trial_seed,
                        }
                        best_run_name = run_name
                except Exception as exc:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    error_path = output_dir / "error.txt"
                    error_path.write_text(traceback.format_exc(), encoding="utf-8")
                    write_run_status(
                        output_dir,
                        {
                            "status": "failed",
                            "model_name": model_name,
                            "config_path": str(Path(args.config).resolve()),
                            "run_name": run_name,
                            "trial_index": trial_index,
                            "trial_seed": trial_seed,
                            "started_at": trial_started_at,
                            "ended_at": current_timestamp(),
                            "duration_seconds": None,
                            "split_rows": {name: len(items) for name, items in splits.items()},
                            "checkpoint_preflight": checkpoint_preflight,
                            "requested_device": args.requested_device,
                            "resolved_device": args.device,
                            "effective_device": args.device,
                            "device_fallback_reason": args.device_fallback_reason,
                            "hyperparameters": hyperparameters,
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                            "traceback_path": str(error_path),
                        },
                    )
                    mlflow.set_tag("trial_status", "failed")
                    mlflow.set_tag("failure_type", type(exc).__name__)
                    mlflow.log_artifact(str(error_path))
                    _log(f"trial failed: {run_name} ({type(exc).__name__}: {exc})")
                finally:
                    if model is not None:
                        del model
                    _cleanup_torch_state(args.device)

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
        mlflow.log_param("run_group_id", args.run_group_id)
        mlflow.log_param("dataset_id", args.dataset_id)
        mlflow.log_param("dataset_spec_path", args.dataset_spec)
        mlflow.log_param("model_family", args.model_family)
        mlflow.log_param("requested_device", args.requested_device)
        mlflow.log_param("resolved_device", args.device)
        mlflow.log_param("effective_device", args.device)
        mlflow.log_param("device_fallback_reason", args.device_fallback_reason)
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


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_run_status(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def flatten_params(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}_{key}"
        if isinstance(value, dict):
            flattened.update(flatten_params(value, prefix=name))
        else:
            flattened[name] = value
    return flattened


def select_trial_score(summary: dict[str, Any]) -> float:
    metrics = summary.get("best_validation_metrics", {})
    for metric_name in [
        PRIMARY_PAUC_METRIC,
        "auc_roc",
        "average_precision",
        "f1_score",
        "balanced_accuracy",
    ]:
        score = float(metrics.get(metric_name, float("nan")))
        if score == score:
            return score
    return float("-inf")


def image_preprocessing_contract(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    configured_mean = dataset_config.get("normalize_mean")
    configured_std = dataset_config.get("normalize_std")
    backend = str(model_config.get("backend", ""))
    display_name = str(model_config.get("display_name", "image_model"))

    if configured_mean is not None or configured_std is not None:
        if configured_mean is None or configured_std is None:
            raise ValueError("Both dataset.normalize_mean and dataset.normalize_std must be set together.")
        mean = [float(value) for value in configured_mean]
        std = [float(value) for value in configured_std]
        source = "config"
        notes = "dataset config normalization override"
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        source = "imagenet_default"
        notes = "ImageNet RGB normalization"

    return {
        "model_name": display_name,
        "backend": backend,
        "normalize_mean": mean,
        "normalize_std": std,
        "source": source,
        "notes": notes,
    }


def _select_trial_plan(
    *,
    combinations: list[dict[str, Any]],
    trial_indices: list[int] | None,
    max_trials: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    indexed_combinations = list(enumerate(combinations))
    if trial_indices is not None and max_trials is not None:
        raise ValueError("`--trial-indices` and `--max-trials` cannot be used together.")
    if trial_indices is None:
        if max_trials is not None:
            return indexed_combinations[: max(max_trials, 0)]
        return indexed_combinations

    if len(set(trial_indices)) != len(trial_indices):
        raise ValueError(f"Duplicate trial indices are not allowed: {trial_indices}")

    selected: list[tuple[int, dict[str, Any]]] = []
    for trial_index in trial_indices:
        if trial_index < 0:
            raise ValueError(f"Trial indices must be non-negative, got {trial_index}")
        if trial_index >= len(indexed_combinations):
            raise ValueError(
                f"Trial index {trial_index} is out of range for {len(indexed_combinations)} available trials."
            )
        selected.append(indexed_combinations[trial_index])
    return selected


def image_split_protocol_audit(
    splits: dict[str, list[Any]],
    *,
    nested_protocol: bool,
) -> dict[str, Any]:
    role_names = (
        {"train": "inner_train", "val": "inner_validation", "test": "outer_test"}
        if nested_protocol
        else {"train": "train", "val": "validation", "test": "test"}
    )
    patient_sets = {
        name: {
            str(sample.metadata.get("patient_id", ""))
            for sample in samples
            if str(sample.metadata.get("patient_id", ""))
        }
        for name, samples in splits.items()
    }
    role_distribution = []
    for split_name in ["train", "val", "test"]:
        samples = splits[split_name]
        patients = patient_sets[split_name]
        positive_rows = sum(int(getattr(sample, "label", 0)) for sample in samples)
        row_count = len(samples)
        role_distribution.append(
            {
                "role": role_names[split_name],
                "rows": int(row_count),
                "patients": int(len(patients)),
                "positive_rows": int(positive_rows),
                "positive_rate_pct": round(float(positive_rows) / max(row_count, 1) * 100, 6),
            }
        )

    if nested_protocol:
        overlap_audit = {
            "inner_train_inner_validation_patient_overlap": len(patient_sets["train"] & patient_sets["val"]),
            "inner_train_outer_test_patient_overlap": len(patient_sets["train"] & patient_sets["test"]),
            "inner_validation_outer_test_patient_overlap": len(patient_sets["val"] & patient_sets["test"]),
        }
    else:
        overlap_audit = {
            "train_val_patient_overlap": len(patient_sets["train"] & patient_sets["val"]),
            "train_test_patient_overlap": len(patient_sets["train"] & patient_sets["test"]),
            "val_test_patient_overlap": len(patient_sets["val"] & patient_sets["test"]),
        }
    failed_checks = {key: value for key, value in overlap_audit.items() if value != 0}
    if failed_checks:
        raise RuntimeError(f"Image split patient overlap audit failed: {failed_checks}")
    return {
        "patient_overlap_audit": overlap_audit,
        "triple_balance_audit": role_distribution,
    }


def patient_balanced_sample_weights(samples: list[Any]) -> list[float]:
    if not samples:
        return []

    patient_counts_by_label: dict[int, dict[str, int]] = {0: {}, 1: {}}
    for sample in samples:
        label = int(getattr(sample, "label", 0))
        patient_key = image_sample_patient_key(sample)
        patient_counts_by_label.setdefault(label, {})
        patient_counts_by_label[label][patient_key] = patient_counts_by_label[label].get(patient_key, 0) + 1

    present_labels = [label for label, patient_counts in patient_counts_by_label.items() if patient_counts]
    if len(present_labels) < 2:
        return [1.0 for _ in samples]

    class_mass = 1.0 / len(present_labels)
    weights = []
    for sample in samples:
        label = int(getattr(sample, "label", 0))
        patient_key = image_sample_patient_key(sample)
        patient_counts = patient_counts_by_label[label]
        weights.append(class_mass / len(patient_counts) / patient_counts[patient_key])
    return weights


def sampler_weight_report(samples: list[Any], weights: list[float], *, strategy: str) -> dict[str, Any]:
    class_weight_sum: dict[int, float] = {0: 0.0, 1: 0.0}
    patient_weight_sum: dict[str, float] = {}
    for sample, weight in zip(samples, weights, strict=True):
        label = int(getattr(sample, "label", 0))
        patient_key = image_sample_patient_key(sample)
        class_weight_sum[label] = class_weight_sum.get(label, 0.0) + float(weight)
        patient_weight_sum[patient_key] = patient_weight_sum.get(patient_key, 0.0) + float(weight)

    return {
        "strategy": strategy,
        "status": "ok",
        "sample_count": len(samples),
        "positive_samples": sum(int(getattr(sample, "label", 0)) for sample in samples),
        "patient_count": len(patient_weight_sum),
        "class_weight_sum": {str(key): value for key, value in sorted(class_weight_sum.items())},
        "min_sample_weight": min(weights) if weights else None,
        "max_sample_weight": max(weights) if weights else None,
    }


def image_sample_patient_key(sample: Any) -> str:
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ["patient_id", "group_id", "isic_id"]:
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value:
            return str(value)
        value = getattr(sample, key, "")
        if value:
            return str(value)
    return str(id(sample))


def _disable_pretrained_weights(config: dict[str, Any]) -> None:
    model_config = config.setdefault("model", {})
    if "weights" in model_config:
        model_config["weights"] = None
    if "pretrained" in model_config:
        model_config["pretrained"] = False
    if "checkpoint_path" in model_config:
        model_config["checkpoint_path"] = None


def _limit_samples(samples: list[Any], max_samples: int | None, seed: int) -> list[Any]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples

    positives = [sample for sample in samples if getattr(sample, "label", 0) == 1]
    negatives = [sample for sample in samples if getattr(sample, "label", 0) == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    if not positives or not negatives:
        limited = samples[:max_samples]
        rng.shuffle(limited)
        return limited

    positive_target = round(max_samples * len(positives) / len(samples))
    positive_target = max(1, min(len(positives), positive_target))
    negative_target = max_samples - positive_target
    negative_target = max(1, min(len(negatives), negative_target))

    if positive_target + negative_target > max_samples:
        overflow = positive_target + negative_target - max_samples
        if negative_target >= positive_target and negative_target - overflow >= 1:
            negative_target -= overflow
        else:
            positive_target = max(1, positive_target - overflow)
    elif positive_target + negative_target < max_samples:
        remaining = max_samples - (positive_target + negative_target)
        extra_negatives = min(remaining, len(negatives) - negative_target)
        negative_target += extra_negatives
        remaining -= extra_negatives
        positive_target += min(remaining, len(positives) - positive_target)

    limited = positives[:positive_target] + negatives[:negative_target]
    rng.shuffle(limited)
    return limited


def _cleanup_torch_state(device: str) -> None:
    gc.collect()
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _validate_runtime_device(device: str, *, requested_device: str | None = None) -> None:
    import torch

    _log(
        "runtime device preflight: "
        f"cuda_available={torch.cuda.is_available()}, visible_device_count={torch.cuda.device_count()}, "
        f"requested_device={requested_device or device}, resolved_device={device}"
    )
    if torch.cuda.is_available():
        visible = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
        _log(f"visible_cuda_devices={json.dumps(visible, ensure_ascii=False)}")

    if not device.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver, CUDA visibility, and conda environment."
        )
    try:
        torch.empty(1, device=device)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise RuntimeError(f"Failed to allocate a tensor on device '{device}': {exc}") from exc


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[run_experiment {timestamp}] {message}", flush=True)


if __name__ == "__main__":
    main()
