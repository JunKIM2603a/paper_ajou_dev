from __future__ import annotations

import argparse
import atexit
import gc
import json
import time
from pathlib import Path
from typing import Any

from isic2024_multimodal.utils.device import resolve_device
from isic2024_multimodal.utils.runtime_env import ensure_expected_conda_env, get_default_mlflow_tracking_uri

DEFAULT_DATASET_ROOT = "data/raw/isic_2024_challenge"
DEFAULT_SEED = 42
DEFAULT_TARGET_COLUMN = "target"
PRIMARY_PAUC_METRIC = "pauc_above_tpr80"
STRICT_BASE = "strict_base"
STRICT_FE = "strict_fe"
STRICT_MAIN_INPUT = "strict_main_input"
DEFAULT_NESTED_SPLIT_CSV = "data/splits/isic2024_official_train_nested_5x4_seed42.csv"
DEFAULT_HOLDOUT_SPLIT_CSV = "data/splits/isic2024_train_validation_test_split_seed42.csv"
DEFAULT_CV_SPLIT_CSV = "data/splits/isic2024_train_validation_5fold_seed42.csv"


def make_run_group_id(prefix: str = "tabular") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def log_event(message: str) -> None:
    print(f"[{current_timestamp()}] [run_tabular_baseline] {message}", flush=True)


def record_timing(timings: dict[str, float], name: str, started_at: float) -> None:
    timings[name] = round(time.time() - started_at, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISIC2024 tabular baseline models.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--eda-dir", default="experiments/evidence/eda/isic_2024")
    parser.add_argument("--feature-set-json", default="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json")
    parser.add_argument("--tracking-uri", default=get_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Baselines")
    parser.add_argument("--output-root", default="experiments/outputs/tabular_baselines")
    parser.add_argument("--run-group-id", default=None, help="Optional run group tag used to scope MLflow reports.")
    parser.add_argument("--dataset-id", default=None, help="Versioned dataset id for registry/report filtering.")
    parser.add_argument("--dataset-spec", default=None, help="Dataset spec JSON path used for this run.")
    parser.add_argument("--model-family", default="tabular_baselines", help="Experiment family tag.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-protocol", choices=["nested_cv", "legacy_holdout"], default="nested_cv")
    parser.add_argument("--nested-split-csv", default=DEFAULT_NESTED_SPLIT_CSV)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--holdout-split-csv", default=DEFAULT_HOLDOUT_SPLIT_CSV)
    parser.add_argument("--cv-split-csv", default=DEFAULT_CV_SPLIT_CSV)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument("--max-val-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate dataset, locked splits, feature sets, devices, and output-root status without training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device for tabular estimators. Use `auto`, `cpu`, or `cuda`; auto prefers CUDA and falls back to CPU.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=[STRICT_BASE, STRICT_FE, STRICT_MAIN_INPUT],
        help=(
            "Subset of feature sets to run. Example: --feature-sets strict_base strict_fe strict_main_input. "
            "strict_main_input is the strict_input contract; relaxed/oracle compatibility keys are not "
            "ordinary inference-time inputs."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "logistic_regression",
            "svm",
            "mlp",
            "xgboost",
            "catboost",
            "lightgbm",
            "ft_transformer",
            "ft_transformer_external",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_group_id = args.run_group_id or make_run_group_id()
    command_start = time.time()
    command_status = {"value": "failed"}

    def log_command_end() -> None:
        log_event(
            f"End status={command_status['value']} "
            f"run_group_id={args.run_group_id} duration={format_duration(time.time() - command_start)}"
        )

    ensure_expected_conda_env()
    requested_device = args.device
    device_resolution = resolve_device(requested_device)
    args.requested_device = device_resolution.requested_device
    args.device = device_resolution.resolved_device
    args.device_fallback_reason = device_resolution.fallback_reason
    args.cuda_available = device_resolution.cuda_available
    args.visible_device_count = device_resolution.visible_device_count
    atexit.register(log_command_end)
    log_event(
        "Start "
        f"run_group_id={args.run_group_id} models={','.join(args.models)} "
        f"feature_sets={','.join(args.feature_sets)} requested_device={args.requested_device} "
        f"resolved_device={args.device} fallback={args.device_fallback_reason or 'none'}"
    )
    global binary_classification_metrics
    global build_catboost_estimator
    global build_final_feature_frames
    global build_lightgbm_estimator
    global build_preprocessor
    global build_sklearn_estimator
    global build_torch_estimator
    global build_xgboost_estimator
    global device_uses_cuda
    global effective_tabular_device
    global expand_search_space
    global get_model_specs
    global is_final_inputs_feature_payload
    global load_tabular_dataframe
    global normalize_feature_set_name
    global normalize_feature_set_names
    global recommend_feature_sets
    global sanitize_run_name
    global set_global_seed
    global split_feature_types
    global select_threshold_by_f1
    global thresholded_binary_classification_metrics

    from isic2024_multimodal.baselines.tabular.baselines import (
        build_catboost_estimator,
        build_lightgbm_estimator,
        build_preprocessor,
        build_sklearn_estimator,
        build_torch_estimator,
        build_xgboost_estimator,
        device_uses_cuda,
        effective_tabular_device,
        get_model_specs,
        split_feature_types,
    )
    from isic2024_multimodal.data.tabular_dataset import load_tabular_dataframe
    from isic2024_multimodal.evaluation.metrics import (
        binary_classification_metrics,
        select_threshold_by_f1,
        thresholded_binary_classification_metrics,
    )
    from isic2024_multimodal.features.final_tabular_inputs import (
        build_final_feature_frames,
        is_final_inputs_feature_payload,
    )
    from isic2024_multimodal.features.tabular_feature_sets import recommend_feature_sets
    from isic2024_multimodal.features.tabular_terms import normalize_feature_set_name, normalize_feature_set_names
    from isic2024_multimodal.training.reproducibility import set_global_seed
    from isic2024_multimodal.utils.config_utils import expand_search_space, sanitize_run_name
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "pandas and mlflow are required to run tabular baselines. Activate the conda env and install dependencies."
        ) from exc

    if args.preflight_only:
        preflight_start = time.time()
        log_event("Start preflight")
        preflight_summary = run_preflight(args)
        log_event(
            f"Finished preflight status={preflight_summary['status']} "
            f"duration={format_duration(time.time() - preflight_start)}"
        )
        print(json.dumps(preflight_summary, ensure_ascii=False, indent=2))
        command_status["value"] = "ok"
        return

    load_start = time.time()
    log_event("Start data_and_protocol_load")
    validate_runtime_device(args.device)
    ensure_feature_set_json(args.eda_dir, args.feature_set_json)
    feature_payload = load_feature_payload(args.feature_set_json)
    available_feature_sets = resolve_available_feature_sets(feature_payload, args.feature_sets, args.feature_set_json)

    set_global_seed(args.seed)
    frame = load_merged_dataframe(args.dataset_root)
    target_column = feature_payload["target_column"]
    if target_column != DEFAULT_TARGET_COLUMN:
        raise RuntimeError(f"Unexpected target column in feature set JSON: {target_column}")
    split_definition = load_split_definition(args)
    split_definition = enrich_split_definition_with_frame_audit(frame, split_definition)
    sample_ids = frame["isic_id"].astype(str).copy()
    use_final_inputs = is_final_inputs_feature_payload(feature_payload)
    final_feature_frames = (
        build_final_feature_frames(frame, args.eda_dir, sorted(available_feature_sets))
        if use_final_inputs
        else {}
    )
    log_event(
        f"Finished data_and_protocol_load rows={len(frame)} "
        f"feature_sets={','.join(sorted(available_feature_sets))} "
        f"duration={format_duration(time.time() - load_start)}"
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    for spec in get_model_specs(args.models):
        model_start = time.time()
        parent_run_name = sanitize_run_name(spec.name)
        model_effective_device = effective_tabular_device(spec.name, args.device)
        combinations = [
            combination
            for combination in expand_search_space(spec.search_space)
            if combination.get("feature_set") in available_feature_sets
        ]
        best_result = None
        best_run_name = None
        log_event(
            f"Start model={spec.name} requested_device={args.requested_device} "
            f"resolved_device={args.device} effective_device={model_effective_device} trials={len(combinations)}"
        )

        with mlflow.start_run(run_name=parent_run_name):
            mlflow.set_tags(
                {
                    "experiment_family": "isic2024_tabular_baseline",
                    "model_name": spec.name,
                    "role": "model_parent",
                    "dataset_name": "ISIC2024-challenge",
                    "run_group_id": args.run_group_id,
                    "dataset_id": args.dataset_id or "",
                    "model_family": args.model_family,
                }
            )
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "split_protocol": args.split_protocol,
                    "outer_fold": args.outer_fold,
                    "inner_fold": args.inner_fold,
                    "nested_split_csv": str(Path(args.nested_split_csv).resolve()),
                    "cv_fold": args.cv_fold,
                    "holdout_split_csv": str(Path(args.holdout_split_csv).resolve()),
                    "cv_split_csv": str(Path(args.cv_split_csv).resolve()),
                    "split_source": split_definition["split_source"],
                    "max_train_rows": args.max_train_rows,
                    "max_val_rows": args.max_val_rows,
                    "max_test_rows": args.max_test_rows,
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "split_group_policy": "patient_id -> lesion_id -> isic_id",
                    "primary_metric_name": PRIMARY_PAUC_METRIC,
                    "threshold_source": threshold_source_for_split(split_definition),
                    "selected_feature_sets": ",".join(sorted(available_feature_sets)),
                    "runtime_device": args.device,
                    "requested_device": args.requested_device,
                    "resolved_device": args.device,
                    "device_fallback_reason": args.device_fallback_reason,
                    "effective_device": model_effective_device,
                    "effective_device_reason": tabular_effective_device_reason(
                        spec.name,
                        resolved_device=args.device,
                        effective_device=model_effective_device,
                    ),
                    "estimator_backend": tabular_estimator_backend(
                        spec.name,
                        resolved_device=args.device,
                        effective_device=model_effective_device,
                    ),
                    "run_group_id": args.run_group_id,
                    "dataset_id": args.dataset_id,
                    "dataset_spec_path": args.dataset_spec,
                    "model_family": args.model_family,
                    **missing_value_policy_params(),
                }
            )

            for index, combination in enumerate(combinations, start=1):
                trial_seed = args.seed + index - 1
                hyperparameters = dict(combination)
                hyperparameters["seed"] = trial_seed
                feature_set_name = normalize_feature_set_name(hyperparameters["feature_set"])
                hyperparameters["feature_set"] = feature_set_name
                if feature_set_name not in available_feature_sets:
                    print(
                        f"[run_tabular_baselines] Skipping {spec.name} / {feature_set_name} "
                        f"because it is not present in {args.feature_set_json}"
                    )
                    continue
                features = feature_payload["feature_sets"][feature_set_name]
                if use_final_inputs:
                    split_frame = final_feature_frames[feature_set_name].copy()
                    missing_features = [column for column in features if column not in split_frame.columns]
                    if missing_features:
                        raise RuntimeError(
                            f"Final-input frame for '{feature_set_name}' is missing columns: {missing_features[:10]}"
                        )
                    split_frame = split_frame[features].copy()
                    split_frame[target_column] = frame[target_column].values
                else:
                    split_frame = frame[features + [target_column]].copy()

                trial_run_name = f"{parent_run_name}_trial_{index:03d}"
                output_dir = Path(args.output_root) / parent_run_name / trial_run_name
                output_dir.mkdir(parents=True, exist_ok=True)
                trial_start = time.time()
                log_event(
                    f"Start trial model={spec.name} trial={index}/{len(combinations)} "
                    f"feature_set={feature_set_name}"
                )
                summary = train_and_evaluate(
                    frame=split_frame,
                    sample_ids=sample_ids,
                    split_definition=split_definition,
                    target_column=target_column,
                    model_name=spec.name,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    device=args.device,
                    requested_device=args.requested_device,
                    device_fallback_reason=args.device_fallback_reason,
                    run_group_id=args.run_group_id,
                    dataset_id=args.dataset_id,
                    dataset_spec_path=args.dataset_spec,
                    model_family=args.model_family,
                    include_test=False,
                    max_train_rows=args.max_train_rows,
                    max_val_rows=args.max_val_rows,
                    max_test_rows=args.max_test_rows,
                )
                log_event(
                    f"Finished trial model={spec.name} trial={index}/{len(combinations)} "
                    f"feature_set={feature_set_name} val_{PRIMARY_PAUC_METRIC}="
                    f"{summary['metrics']['val'][PRIMARY_PAUC_METRIC]:.6f} "
                    f"duration={format_duration(time.time() - trial_start)}"
                )

                with mlflow.start_run(run_name=trial_run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "experiment_family": "isic2024_tabular_baseline",
                            "model_name": spec.name,
                            "role": "hyperparameter_trial",
                            "feature_set": feature_set_name,
                            "run_group_id": args.run_group_id,
                            "dataset_id": args.dataset_id or "",
                            "model_family": args.model_family,
                        }
                    )
                    mlflow.log_params({f"hp_{key}": normalize_param_value(value) for key, value in hyperparameters.items()})
                    mlflow.log_param("feature_count", len(features))
                    mlflow.log_param("feature_set", feature_set_name)
                    mlflow.log_param("runtime_device", args.device)
                    mlflow.log_param("requested_device", args.requested_device)
                    mlflow.log_param("resolved_device", args.device)
                    mlflow.log_param("device_fallback_reason", args.device_fallback_reason)
                    mlflow.log_param("effective_device", summary["effective_device"])
                    mlflow.log_param("estimator_backend", summary["estimator_backend"])
                    mlflow.log_param("run_group_id", args.run_group_id)
                    mlflow.log_param("dataset_id", args.dataset_id)
                    mlflow.log_param("dataset_spec_path", args.dataset_spec)
                    mlflow.log_param("model_family", args.model_family)
                    mlflow.log_params(
                        {
                            f"runtime_{key}": normalize_param_value(value)
                            for key, value in summary["runtime_parameters"].items()
                        }
                    )
                    mlflow.log_params(missing_value_policy_params(summary["missing_value_policy"]))
                    mlflow.log_dict(summary["split_summary"], "split_summary.json")
                    mlflow.log_dict(summary["metrics"], "metrics.json")
                    mlflow.log_dict(summary["hyperparameters"], "hyperparameters.json")
                    for metric_group, metrics in summary["metrics"].items():
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(f"{metric_group}_{metric_name}", float(metric_value))
                    mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
                    for timing_name, timing_value in summary["timing_seconds"].items():
                        mlflow.log_metric(f"timing_{timing_name}", float(timing_value))
                    mlflow.log_artifact(str(output_dir / "summary.json"))

                score = select_trial_score(summary)
                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "summary": summary,
                        "hyperparameters": hyperparameters,
                        "feature_set_name": feature_set_name,
                        "features": features,
                    }
                    best_run_name = trial_run_name

            if best_result is None:
                raise RuntimeError(f"No successful tabular trials completed for {spec.name}")

            best_feature_set_name = best_result["feature_set_name"]
            best_features = best_result["features"]
            if use_final_inputs:
                best_split_frame = final_feature_frames[best_feature_set_name][best_features].copy()
                best_split_frame[target_column] = frame[target_column].values
            else:
                best_split_frame = frame[best_features + [target_column]].copy()
            final_output_dir = Path(args.output_root) / parent_run_name / "best_final_test"
            final_start = time.time()
            log_event(
                f"Start final_test model={spec.name} best_feature_set={best_feature_set_name}"
            )
            best_final_summary = train_and_evaluate(
                frame=best_split_frame,
                sample_ids=sample_ids,
                split_definition=split_definition,
                target_column=target_column,
                model_name=spec.name,
                hyperparameters=best_result["hyperparameters"],
                output_dir=final_output_dir,
                device=args.device,
                requested_device=args.requested_device,
                device_fallback_reason=args.device_fallback_reason,
                run_group_id=args.run_group_id,
                dataset_id=args.dataset_id,
                dataset_spec_path=args.dataset_spec,
                model_family=args.model_family,
                include_test=True,
                max_train_rows=args.max_train_rows,
                max_val_rows=args.max_val_rows,
                max_test_rows=args.max_test_rows,
            )
            log_event(
                f"Finished final_test model={spec.name} test_{PRIMARY_PAUC_METRIC}="
                f"{best_final_summary['metrics']['test'][PRIMARY_PAUC_METRIC]:.6f} "
                f"duration={format_duration(time.time() - final_start)}"
            )

            mlflow.log_dict(best_final_summary, "best_summary.json")
            mlflow.set_tag("best_child_run_name", best_run_name)
            mlflow.log_params(
                {
                    f"best_hp_{key}": normalize_param_value(value)
                    for key, value in best_result["hyperparameters"].items()
                }
            )
            mlflow.log_param("best_feature_set", best_result["hyperparameters"]["feature_set"])
            mlflow.log_param("run_group_id", args.run_group_id)
            mlflow.log_param("dataset_id", args.dataset_id)
            mlflow.log_param("dataset_spec_path", args.dataset_spec)
            mlflow.log_param("model_family", args.model_family)
            mlflow.log_param("selected_threshold", best_final_summary["selected_threshold"])
            mlflow.log_param("threshold_source", best_final_summary["threshold_source"])
            mlflow.log_param("requested_device", args.requested_device)
            mlflow.log_param("resolved_device", args.device)
            mlflow.log_param("device_fallback_reason", args.device_fallback_reason)
            mlflow.log_param("best_effective_device", best_final_summary["effective_device"])
            mlflow.log_param("best_estimator_backend", best_final_summary["estimator_backend"])
            mlflow.log_params(
                {
                    f"best_runtime_{key}": normalize_param_value(value)
                    for key, value in best_final_summary["runtime_parameters"].items()
                }
            )
            for metric_name, metric_value in best_final_summary["metrics"]["test"].items():
                mlflow.log_metric(f"best_{metric_name}", float(metric_value))
            mlflow.log_metric("best_duration_seconds", float(best_final_summary["duration_seconds"]))
            for timing_name, timing_value in best_final_summary["timing_seconds"].items():
                mlflow.log_metric(f"best_timing_{timing_name}", float(timing_value))
        log_event(
            f"Finished model={spec.name} best_feature_set={best_result['feature_set_name']} "
            f"duration={format_duration(time.time() - model_start)}"
        )

    command_status["value"] = "ok"


def ensure_feature_set_json(eda_dir: str, feature_set_json: str) -> None:
    path = Path(feature_set_json)
    if path.exists():
        return
    try:
        payload = recommend_feature_sets(eda_dir)
    except FileNotFoundError:
        from isic2024_multimodal.cli.export_strict_input_dataset import STRICT_INPUT_COLUMNS

        payload = {
            "target_column": DEFAULT_TARGET_COLUMN,
            "feature_sets": {
                STRICT_MAIN_INPUT: STRICT_INPUT_COLUMNS,
            },
            "feature_set_aliases": {
                "strict": STRICT_MAIN_INPUT,
                STRICT_MAIN_INPUT: STRICT_MAIN_INPUT,
            },
            "rationales": {
                STRICT_MAIN_INPUT: [
                    "Fallback feature payload from export_strict_input_dataset strict input contract.",
                    "Generate notebook-derived final_inputs evidence before paper-facing strict_base/strict_fe comparisons.",
                ],
            },
            "evidence": {
                "feature_sets_source": "export_strict_input_dataset.STRICT_INPUT_COLUMNS",
                "paper_valid_scope": "strict_main_input smoke/baseline only",
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_payload(feature_set_json: str) -> dict[str, Any]:
    from isic2024_multimodal.features.tabular_terms import normalize_feature_set_name

    path = Path(feature_set_json)
    if not path.exists():
        raise FileNotFoundError(f"Feature set JSON not found: {path}")
    feature_payload = json.loads(path.read_text(encoding="utf-8"))
    raw_feature_sets = feature_payload.get("feature_sets", {})
    feature_payload["feature_sets"] = {
        normalize_feature_set_name(name): columns for name, columns in raw_feature_sets.items()
    }
    return feature_payload


def resolve_available_feature_sets(
    feature_payload: dict[str, Any],
    requested_feature_sets: list[str] | None,
    feature_set_json: str,
) -> set[str]:
    from isic2024_multimodal.features.tabular_terms import normalize_feature_set_names

    all_feature_sets = set(feature_payload["feature_sets"].keys())
    if not all_feature_sets:
        raise RuntimeError(f"No feature sets were found in {feature_set_json}")
    normalized_requested = normalize_feature_set_names(requested_feature_sets)
    if normalized_requested:
        unknown_feature_sets = sorted(set(normalized_requested) - all_feature_sets)
        if unknown_feature_sets:
            raise RuntimeError(
                f"Requested feature sets are not available in {feature_set_json}: {unknown_feature_sets}"
            )
        return set(normalized_requested)
    return set(all_feature_sets)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from isic2024_multimodal.baselines.tabular.baselines import effective_tabular_device, get_model_specs
    from isic2024_multimodal.features.final_tabular_inputs import (
        build_final_feature_frames,
        is_final_inputs_feature_payload,
    )

    validate_runtime_device(args.device)
    feature_payload = load_feature_payload(args.feature_set_json)
    available_feature_sets = resolve_available_feature_sets(feature_payload, args.feature_sets, args.feature_set_json)
    specs = get_model_specs(args.models)
    frame = load_merged_dataframe(args.dataset_root)
    split_definition = load_split_definition(args)
    split_definition = enrich_split_definition_with_frame_audit(frame, split_definition)
    sample_ids = set(frame["isic_id"].astype(str))
    split_ids = split_definition["train_ids"] | split_definition["val_ids"] | split_definition["test_ids"]
    missing_split_ids = sorted(split_ids - sample_ids)
    if missing_split_ids:
        raise RuntimeError(
            "Locked split IDs are missing from the dataset frame. "
            f"Missing count={len(missing_split_ids)}, examples={missing_split_ids[:10]}"
        )

    feature_missing: dict[str, list[str]] = {}
    if is_final_inputs_feature_payload(feature_payload):
        final_feature_frames = build_final_feature_frames(frame, args.eda_dir, sorted(available_feature_sets))
        for feature_set_name in sorted(available_feature_sets):
            columns = set(final_feature_frames[feature_set_name].columns)
            missing = [column for column in feature_payload["feature_sets"][feature_set_name] if column not in columns]
            if missing:
                feature_missing[feature_set_name] = missing[:10]
    else:
        frame_columns = set(frame.columns)
        for feature_set_name in sorted(available_feature_sets):
            missing = [column for column in feature_payload["feature_sets"][feature_set_name] if column not in frame_columns]
            if missing:
                feature_missing[feature_set_name] = missing[:10]
    if feature_missing:
        raise RuntimeError(f"Feature columns are missing from the preflight frame: {feature_missing}")

    output_root = Path(args.output_root)
    summary_count = len(list(output_root.glob("**/summary.json"))) if output_root.exists() else 0
    output_warnings = []
    if summary_count:
        output_warnings.append(
            f"Output root already contains {summary_count} summary.json files; use a separate smoke output root when row caps are set."
        )

    return {
        "status": "ok",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_rows": int(len(frame)),
        "positive_rows": int(frame[DEFAULT_TARGET_COLUMN].sum()),
        "feature_set_json": str(Path(args.feature_set_json).resolve()),
        "feature_sets": sorted(available_feature_sets),
        "models": [spec.name for spec in specs],
        "run_group_id": getattr(args, "run_group_id", None),
        "dataset_id": getattr(args, "dataset_id", None),
        "dataset_spec_path": getattr(args, "dataset_spec", None),
        "model_family": getattr(args, "model_family", "tabular_baselines"),
        "requested_device": getattr(args, "requested_device", args.device),
        "resolved_device": args.device,
        "cuda_available": getattr(args, "cuda_available", None),
        "visible_device_count": getattr(args, "visible_device_count", None),
        "device_fallback_reason": getattr(args, "device_fallback_reason", None),
        "effective_devices": {spec.name: effective_tabular_device(spec.name, args.device) for spec in specs},
        "effective_device_reasons": {
            spec.name: tabular_effective_device_reason(
                spec.name,
                resolved_device=args.device,
                effective_device=effective_tabular_device(spec.name, args.device),
            )
            for spec in specs
        },
        "split_protocol": getattr(args, "split_protocol", "nested_cv"),
        "split_source": split_definition["split_source"],
        "nested_split_csv": split_definition.get("nested_split_csv"),
        "outer_fold": split_definition.get("outer_fold"),
        "inner_fold": split_definition.get("inner_fold"),
        "split_rows": {
            "train": int(split_definition["num_train_rows"]),
            "val": int(split_definition["num_val_rows"]),
            "test": int(split_definition["num_test_rows"]),
        },
        "overlap_checks": split_definition["overlap_checks"],
        "triple_balance_audit": split_definition.get("triple_balance_audit"),
        "output_root": str(output_root.resolve()),
        "output_root_exists": output_root.exists(),
        "output_summary_count": summary_count,
        "output_warnings": output_warnings,
    }


def load_merged_dataframe(dataset_root: str):
    from isic2024_multimodal.data.tabular_dataset import load_tabular_dataframe

    frame = load_tabular_dataframe(dataset_root, include_image_columns=False)
    if DEFAULT_TARGET_COLUMN not in frame.columns:
        raise RuntimeError(f"Target column '{DEFAULT_TARGET_COLUMN}' not found in {dataset_root}")
    return frame


def select_trial_score(summary: dict[str, Any]) -> float:
    val_metrics = summary["metrics"]["val"]
    score = val_metrics[PRIMARY_PAUC_METRIC]
    if score != score:
        score = val_metrics["auc_roc"]
    if score != score:
        score = val_metrics["average_precision"]
    return float(score)


def tabular_estimator_backend(model_name: str, *, resolved_device: str, effective_device: str) -> str:
    if model_name in {"logistic_regression", "svm", "mlp"}:
        return "repo_torch" if str(effective_device).startswith("cuda") else "sklearn"
    if model_name in {"ft_transformer", "ft_transformer_external"}:
        return "repo_torch"
    if model_name == "xgboost":
        return "xgboost_cuda" if str(effective_device).startswith("cuda") else "xgboost_cpu"
    if model_name == "catboost":
        return "catboost_gpu" if str(effective_device).startswith("cuda") else "catboost_cpu"
    if model_name == "lightgbm":
        return "lightgbm_cpu"
    return f"{model_name}_{effective_device or resolved_device}"


def tabular_effective_device_reason(model_name: str, *, resolved_device: str, effective_device: str) -> str | None:
    if model_name == "lightgbm" and str(resolved_device).startswith("cuda") and effective_device == "cpu":
        return "LightGBM GPU backend uses OpenCL and is not validated in this CUDA workflow; using CPU."
    if (
        model_name in {"logistic_regression", "svm", "mlp"}
        and str(effective_device).startswith("cuda")
    ):
        return "CUDA request uses the repo-native torch estimator, not the sklearn estimator."
    if str(resolved_device).startswith("cuda") and effective_device == "cpu":
        return f"{model_name} is not marked as CUDA-capable; using CPU."
    return None


def train_and_evaluate(
    *,
    frame,
    sample_ids,
    split_definition: dict[str, Any],
    target_column: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    output_dir: Path,
    device: str,
    requested_device: str,
    device_fallback_reason: str | None,
    run_group_id: str,
    dataset_id: str | None,
    dataset_spec_path: str | None,
    model_family: str,
    include_test: bool,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, Any]:
    from isic2024_multimodal.baselines.tabular.baselines import effective_tabular_device
    from isic2024_multimodal.features.tabular_missing import missing_value_policy_summary

    start = time.time()
    started_at = current_timestamp()
    timings: dict[str, float] = {}
    stage_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(hyperparameters["seed"])
    set_global_seed(seed)
    effective_device = effective_tabular_device(model_name, device)
    estimator_device = "cpu" if effective_device == "cpu" else device
    estimator = None
    record_timing(timings, "initialize_seconds", stage_start)

    try:
        stage_start = time.time()
        X = frame.drop(columns=[target_column]).copy()
        y = frame[target_column].astype(float).astype(int)

        for column in X.columns:
            X[column] = X[column].where(X[column].notna(), None)

        sample_ids = sample_ids.astype(str)
        train_mask = sample_ids.isin(split_definition["train_ids"])
        val_mask = sample_ids.isin(split_definition["val_ids"])
        test_mask = sample_ids.isin(split_definition["test_ids"])
        X_train = X.loc[train_mask].copy()
        X_val = X.loc[val_mask].copy()
        y_train = y.loc[train_mask].copy()
        y_val = y.loc[val_mask].copy()
        if include_test:
            X_test = X.loc[test_mask].copy()
            y_test = y.loc[test_mask].copy()

        X_train, y_train = limit_split_rows(X_train, y_train, max_train_rows, seed=seed)
        X_val, y_val = limit_split_rows(X_val, y_val, max_val_rows, seed=seed + 1)
        if include_test:
            X_test, y_test = limit_split_rows(X_test, y_test, max_test_rows, seed=seed + 2)

        numeric_columns, categorical_columns = split_feature_types(X_train)
        record_timing(timings, "prepare_splits_seconds", stage_start)

        stage_start = time.time()
        estimator = build_estimator(
            model_name=model_name,
            hyperparameters=hyperparameters,
            X_train=X_train,
            y_train=y_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            device=estimator_device,
        )
        record_timing(timings, "build_estimator_seconds", stage_start)

        stage_start = time.time()
        estimator.fit(X_train, y_train)
        record_timing(timings, "fit_seconds", stage_start)
        runtime_parameters = estimator.runtime_params() if hasattr(estimator, "runtime_params") else {}

        stage_start = time.time()
        val_labels, val_probabilities = predict_probabilities(estimator, X_val, y_val)
        selected_threshold = select_threshold_by_f1(val_labels, val_probabilities)
        record_timing(timings, "select_threshold_seconds", stage_start)

        stage_start = time.time()
        train_metrics = evaluate_predictions(estimator, X_train, y_train, threshold=selected_threshold)
        record_timing(timings, "evaluate_train_seconds", stage_start)

        stage_start = time.time()
        val_metrics = evaluate_predictions(estimator, X_val, y_val, threshold=selected_threshold)
        record_timing(timings, "evaluate_val_seconds", stage_start)

        metrics = {
            "train": train_metrics,
            "val": val_metrics,
        }
        if include_test:
            stage_start = time.time()
            metrics["test"] = evaluate_predictions(estimator, X_test, y_test, threshold=selected_threshold)
            record_timing(timings, "evaluate_test_seconds", stage_start)
        duration_seconds = time.time() - start
        timings["total_seconds"] = round(duration_seconds, 6)

        summary = {
            "model_name": model_name,
            "hyperparameters": {key: normalize_param_value(value) for key, value in hyperparameters.items()},
            "run_group_id": run_group_id,
            "dataset_id": dataset_id,
            "dataset_spec_path": dataset_spec_path,
            "model_family": model_family,
            "started_at": started_at,
            "ended_at": current_timestamp(),
            "requested_device": requested_device,
            "resolved_device": device,
            "device_fallback_reason": device_fallback_reason,
            "effective_device": effective_device,
            "effective_device_reason": tabular_effective_device_reason(
                model_name,
                resolved_device=device,
                effective_device=effective_device,
            ),
            "estimator_backend": tabular_estimator_backend(
                model_name,
                resolved_device=device,
                effective_device=effective_device,
            ),
            "runtime_parameters": runtime_parameters,
            "threshold_source": threshold_source_for_split(split_definition),
            "selected_threshold": selected_threshold,
            "split_source": split_definition["split_source"],
            "split_protocol": split_definition.get("split_protocol", "legacy_holdout"),
            "missing_value_policy": missing_value_policy_summary(),
            "split_summary": {
                "num_train_rows": int(len(X_train)),
                "num_val_rows": int(len(X_val)),
                "num_test_rows": int(len(X_test) if include_test else split_definition["num_test_rows"]),
                "num_train_positive": int(y_train.sum()),
                "num_val_positive": int(y_val.sum()),
                "num_test_positive": int(y_test.sum() if include_test else y.loc[test_mask].sum()),
                "num_train_groups": int(split_definition["num_train_patients"]),
                "num_val_groups": int(split_definition["num_val_patients"]),
                "num_test_groups": int(split_definition["num_test_patients"]),
                "locked_num_train_rows": int(split_definition["num_train_rows"]),
                "locked_num_val_rows": int(split_definition["num_val_rows"]),
                "locked_num_test_rows": int(split_definition["num_test_rows"]),
                "split_group_policy": "patient_id -> lesion_id -> isic_id",
                "nested_split_csv": split_definition.get("nested_split_csv"),
                "outer_fold": split_definition.get("outer_fold"),
                "inner_fold": split_definition.get("inner_fold"),
                "holdout_split_csv": split_definition.get("holdout_split_csv"),
                "cv_split_csv": split_definition.get("cv_split_csv"),
                "cv_fold": split_definition.get("cv_fold"),
                "threshold_source": threshold_source_for_split(split_definition),
                "patient_overlap_audit": split_definition["overlap_checks"],
                "triple_balance_audit": split_definition.get("triple_balance_audit"),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "runtime_device": device,
                "requested_device": requested_device,
                "resolved_device": device,
                "effective_device": effective_device,
                "device_fallback_reason": device_fallback_reason,
                "estimator_backend": tabular_estimator_backend(
                    model_name,
                    resolved_device=device,
                    effective_device=effective_device,
                ),
                "max_train_rows": max_train_rows,
                "max_val_rows": max_val_rows,
                "max_test_rows": max_test_rows,
            },
            "metrics": metrics,
            "duration_seconds": duration_seconds,
            "timing_seconds": timings,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        estimator = None
        cleanup_runtime_resources(estimator_device)


def limit_split_rows(X, y, max_rows: int | None, *, seed: int):
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y

    import pandas as pd

    positive_index = y.loc[y.astype(int).eq(1)].index.to_list()
    negative_index = y.loc[y.astype(int).eq(0)].index.to_list()
    rng = __import__("random").Random(seed)
    rng.shuffle(positive_index)
    rng.shuffle(negative_index)
    if positive_index and negative_index:
        positive_target = max(1, round(max_rows * len(positive_index) / len(y)))
        positive_target = min(positive_target, len(positive_index))
        negative_target = max_rows - positive_target
        negative_target = max(1, min(negative_target, len(negative_index)))
        selected_index = positive_index[:positive_target] + negative_index[:negative_target]
    else:
        selected_index = list(X.index[:max_rows])
    selected_index = pd.Index(selected_index)
    return X.loc[selected_index].copy(), y.loc[selected_index].copy()


def load_split_definition(args: argparse.Namespace) -> dict[str, Any]:
    default_protocol = "nested_cv" if hasattr(args, "nested_split_csv") else "legacy_holdout"
    if getattr(args, "split_protocol", default_protocol) == "legacy_holdout":
        return load_locked_split_definition(
            holdout_split_csv=args.holdout_split_csv,
            cv_split_csv=args.cv_split_csv,
            cv_fold=args.cv_fold,
        )
    return load_nested_split_definition(
        nested_split_csv=args.nested_split_csv,
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
    )


def threshold_source_for_split(split_definition: dict[str, Any]) -> str:
    return "inner_validation_f1" if split_definition.get("split_protocol") == "nested_cv" else "validation_f1"


def enrich_split_definition_with_frame_audit(frame, split_definition: dict[str, Any]) -> dict[str, Any]:
    split_definition = dict(split_definition)
    split_definition["triple_balance_audit"] = build_tabular_split_balance_audit(frame, split_definition)
    return split_definition


def build_tabular_split_balance_audit(frame, split_definition: dict[str, Any]) -> list[dict[str, Any]]:
    from isic2024_multimodal.data.triple_stratified_split import make_patient_split_profile

    sample_ids = frame["isic_id"].astype(str)
    patient_profile = make_patient_split_profile(frame)
    roles = [
        ("inner_train" if split_definition.get("split_protocol") == "nested_cv" else "train", split_definition["train_ids"]),
        (
            "inner_validation" if split_definition.get("split_protocol") == "nested_cv" else "validation",
            split_definition["val_ids"],
        ),
        ("outer_test" if split_definition.get("split_protocol") == "nested_cv" else "test", split_definition["test_ids"]),
    ]
    rows: list[dict[str, Any]] = []
    for role, role_ids in roles:
        role_mask = sample_ids.isin(role_ids)
        role_frame = frame.loc[role_mask]
        role_patient_ids = set(role_frame["patient_id"].astype(str))
        role_profile = patient_profile.loc[patient_profile["patient_id"].astype(str).isin(role_patient_ids)]
        row_count = int(len(role_frame))
        rows.append(
            {
                "role": role,
                "rows": row_count,
                "patients": int(len(role_patient_ids)),
                "positive_rows": int(role_frame[DEFAULT_TARGET_COLUMN].sum()),
                "positive_rate_pct": round(
                    float(role_frame[DEFAULT_TARGET_COLUMN].sum()) / max(row_count, 1) * 100,
                    6,
                ),
                "malignant_patients": int(role_profile["has_malignant"].sum()),
                "sample_count_bins_present": int(role_profile["sample_count_bin"].nunique()),
                "triple_strata_present": int(role_profile["triple_stratum"].nunique()),
            }
        )
    return rows


def load_nested_split_definition(*, nested_split_csv: str, outer_fold: int, inner_fold: int) -> dict[str, Any]:
    import pandas as pd

    nested_path = Path(nested_split_csv)
    if not nested_path.exists():
        raise FileNotFoundError(
            "Nested CV split CSV is required for paper-valid tabular baselines. "
            f"Missing: {nested_path}. Generate it with: "
            "`conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src "
            "python -m isic2024_multimodal.cli.export_strict_input_dataset`"
        )

    split_frame = pd.read_csv(nested_path, low_memory=False)
    required_columns = {"isic_id", "patient_id", "outer_fold", "inner_fold", "split_role"}
    if not required_columns.issubset(split_frame.columns):
        raise RuntimeError(f"Nested split CSV is missing columns: {sorted(required_columns - set(split_frame.columns))}")

    split_frame["isic_id"] = split_frame["isic_id"].astype(str)
    split_frame["patient_id"] = split_frame["patient_id"].astype(str)
    split_frame["outer_fold"] = split_frame["outer_fold"].astype(int)
    split_frame["inner_fold"] = split_frame["inner_fold"].astype(int)
    selected = split_frame.loc[
        split_frame["outer_fold"].eq(int(outer_fold))
        & split_frame["inner_fold"].eq(int(inner_fold))
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No nested split rows found for outer_fold={outer_fold}, inner_fold={inner_fold}")
    if not selected["isic_id"].is_unique:
        duplicate_count = int(selected["isic_id"].duplicated().sum())
        raise RuntimeError(
            "Nested split must have exactly one role row per isic_id for the selected "
            f"outer/inner fold, duplicate_count={duplicate_count}"
        )
    allowed_roles = {"inner_train", "inner_validation", "outer_test"}
    unexpected_roles = sorted(set(selected["split_role"].astype(str)) - allowed_roles)
    if unexpected_roles:
        raise RuntimeError(f"Nested split CSV has unexpected split_role values: {unexpected_roles}")

    train_frame = selected.loc[selected["split_role"].eq("inner_train")].copy()
    val_frame = selected.loc[selected["split_role"].eq("inner_validation")].copy()
    test_frame = selected.loc[selected["split_role"].eq("outer_test")].copy()
    train_ids = set(train_frame["isic_id"])
    val_ids = set(val_frame["isic_id"])
    test_ids = set(test_frame["isic_id"])
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "Nested split produced an empty tabular split: "
            f"inner_train={len(train_ids)}, inner_validation={len(val_ids)}, outer_test={len(test_ids)}"
        )

    train_patients = set(train_frame["patient_id"])
    val_patients = set(val_frame["patient_id"])
    test_patients = set(test_frame["patient_id"])
    overlap_checks = {
        "inner_train_inner_validation_patient_overlap": len(train_patients & val_patients),
        "inner_train_outer_test_patient_overlap": len(train_patients & test_patients),
        "inner_validation_outer_test_patient_overlap": len(val_patients & test_patients),
    }
    failed_checks = {key: value for key, value in overlap_checks.items() if value != 0}
    if failed_checks:
        raise RuntimeError(f"Nested split patient overlap audit failed: {failed_checks}")

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "split_protocol": "nested_cv",
        "split_source": "nested_cv_split_csv",
        "nested_split_csv": str(nested_path),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "num_train_rows": len(train_ids),
        "num_val_rows": len(val_ids),
        "num_test_rows": len(test_ids),
        "num_train_patients": len(train_patients),
        "num_val_patients": len(val_patients),
        "num_test_patients": len(test_patients),
        "overlap_checks": overlap_checks,
    }


def load_locked_split_definition(*, holdout_split_csv: str, cv_split_csv: str, cv_fold: int) -> dict[str, Any]:
    import pandas as pd

    holdout_path = Path(holdout_split_csv)
    cv_path = Path(cv_split_csv)
    if not holdout_path.exists() or not cv_path.exists():
        missing_paths = [str(path) for path in [holdout_path, cv_path] if not path.exists()]
        raise FileNotFoundError(
            "Locked split CSV files are required for paper-valid tabular baselines. "
            f"Missing: {missing_paths}. Generate them with: "
            "`conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src "
            "python -m isic2024_multimodal.cli.export_strict_input_dataset`"
        )

    holdout_frame = pd.read_csv(holdout_path, low_memory=False)
    cv_frame = pd.read_csv(cv_path, low_memory=False)
    required_holdout_columns = {"isic_id", "patient_id", "split"}
    required_cv_columns = {"isic_id", "patient_id", "cv_validation_fold"}
    if not required_holdout_columns.issubset(holdout_frame.columns):
        raise RuntimeError(f"Holdout split CSV is missing columns: {sorted(required_holdout_columns - set(holdout_frame.columns))}")
    if not required_cv_columns.issubset(cv_frame.columns):
        raise RuntimeError(f"CV split CSV is missing columns: {sorted(required_cv_columns - set(cv_frame.columns))}")

    holdout_frame["isic_id"] = holdout_frame["isic_id"].astype(str)
    holdout_frame["patient_id"] = holdout_frame["patient_id"].astype(str)
    cv_frame["isic_id"] = cv_frame["isic_id"].astype(str)
    cv_frame["patient_id"] = cv_frame["patient_id"].astype(str)
    cv_frame["cv_validation_fold"] = cv_frame["cv_validation_fold"].astype(int)

    train_validation_frame = holdout_frame.loc[holdout_frame["split"].eq("train_validation_data")].copy()
    test_frame = holdout_frame.loc[holdout_frame["split"].eq("test_data")].copy()
    val_frame = cv_frame.loc[cv_frame["cv_validation_fold"].eq(int(cv_fold))].copy()
    if val_frame.empty:
        raise RuntimeError(f"No rows found for cv_fold={cv_fold} in {cv_split_csv}")

    train_validation_ids = set(train_validation_frame["isic_id"])
    val_ids = set(val_frame["isic_id"])
    train_ids = train_validation_ids - val_ids
    test_ids = set(test_frame["isic_id"])
    train_patients = set(train_validation_frame.loc[train_validation_frame["isic_id"].isin(train_ids), "patient_id"])
    val_patients = set(val_frame["patient_id"])
    test_patients = set(test_frame["patient_id"])
    overlap_checks = {
        "train_val_patient_overlap": len(train_patients & val_patients),
        "train_test_patient_overlap": len(train_patients & test_patients),
        "val_test_patient_overlap": len(val_patients & test_patients),
    }
    failed_checks = {key: value for key, value in overlap_checks.items() if value != 0}
    if failed_checks:
        raise RuntimeError(f"Locked split patient overlap audit failed: {failed_checks}")

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "split_protocol": "legacy_holdout",
        "split_source": "locked_split_csv",
        "holdout_split_csv": str(holdout_path),
        "cv_split_csv": str(cv_path),
        "cv_fold": int(cv_fold),
        "num_train_rows": len(train_ids),
        "num_val_rows": len(val_ids),
        "num_test_rows": len(test_ids),
        "num_train_patients": len(train_patients),
        "num_val_patients": len(val_patients),
        "num_test_patients": len(test_patients),
        "overlap_checks": overlap_checks,
    }


def build_estimator(
    *,
    model_name: str,
    hyperparameters: dict[str, Any],
    X_train,
    y_train,
    numeric_columns,
    categorical_columns,
    device: str,
):
    from isic2024_multimodal.features.tabular_missing import CatBoostMissingValuePreprocessor

    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int(len(y_train) - positive_count), 1)
    scale_pos_weight = negative_count / positive_count

    if model_name in {"logistic_regression", "svm", "mlp", "ft_transformer", "ft_transformer_external"} and (
        device_uses_cuda(device) or model_name in {"ft_transformer", "ft_transformer_external"}
    ):
        return build_torch_estimator(
            model_name,
            hyperparameters,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            scale_pos_weight=scale_pos_weight,
            device=device,
        )

    if model_name in {"logistic_regression", "svm", "mlp"}:
        from sklearn.pipeline import Pipeline

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        estimator = build_sklearn_estimator(model_name, hyperparameters)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    if model_name == "xgboost":
        from sklearn.pipeline import Pipeline

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        estimator = build_xgboost_estimator(hyperparameters, scale_pos_weight=scale_pos_weight, device=device)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    if model_name == "lightgbm":
        from sklearn.pipeline import Pipeline

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        estimator = build_lightgbm_estimator(hyperparameters, scale_pos_weight=scale_pos_weight, device=device)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    if model_name == "catboost":
        preprocessor = CatBoostMissingValuePreprocessor(
            numeric_columns=list(numeric_columns),
            categorical_columns=list(categorical_columns),
        )
        estimator = build_catboost_estimator(hyperparameters, device=device)
        return CatBoostWrapper(estimator=estimator, preprocessor=preprocessor, categorical_columns=categorical_columns)

    raise ValueError(f"Unsupported tabular model: {model_name}")


class CatBoostWrapper:
    def __init__(self, *, estimator, preprocessor, categorical_columns: list[str]) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.categorical_columns = categorical_columns

    def fit(self, X, y):
        train_frame = self.preprocessor.fit_transform(X)
        self.estimator.fit(train_frame, y, cat_features=self.categorical_columns)
        return self

    def _prepare(self, X):
        return self.preprocessor.transform(X)

    def predict(self, X):
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._prepare(X))


def predict_probabilities(estimator, X, y_true) -> tuple[list[int], list[float]]:
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    else:
        y_score = estimator.predict(X)

    labels = [int(value) for value in y_true.tolist()]
    probabilities = [float(value) for value in list(y_score)]
    return labels, probabilities


def evaluate_predictions(estimator, X, y_true, *, threshold: float) -> dict[str, float]:
    labels, probabilities = predict_probabilities(estimator, X, y_true)
    return thresholded_binary_classification_metrics(labels, probabilities, threshold=threshold)


def normalize_param_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def missing_value_policy_params(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    from isic2024_multimodal.features.tabular_missing import missing_value_policy_summary

    payload = policy or missing_value_policy_summary()
    params = {}
    for key, value in payload.items():
        if isinstance(value, list):
            params[key] = ",".join(str(item) for item in value)
        else:
            params[key] = value
    return params


def cleanup_runtime_resources(device: str) -> None:
    gc.collect()
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def validate_runtime_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU device was requested for tabular baselines, but torch.cuda.is_available() is False."
        )
    try:
        torch.empty(1, device=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize tabular runtime device '{device}': {exc}") from exc


if __name__ == "__main__":
    main()
