from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from isic2024_multimodal.utils.runtime_env import ensure_expected_conda_env, get_default_mlflow_tracking_uri

DEFAULT_DATASET_ROOT = "data/raw/isic_2024_challenge"
DEFAULT_SEED = 42
DEFAULT_TARGET_COLUMN = "target"
PRIMARY_PAUC_METRIC = "pauc_above_tpr80"
STRICT_BASE = "strict_base"
STRICT_FE = "strict_fe"
STRICT_MAIN_INPUT = "strict_main_input"
DEFAULT_HOLDOUT_SPLIT_CSV = "data/splits/isic2024_train_validation_test_split_seed42.csv"
DEFAULT_CV_SPLIT_CSV = "data/splits/isic2024_train_validation_5fold_seed42.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISIC2024 tabular baseline models.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--eda-dir", default="experiments/evidence/eda/isic_2024")
    parser.add_argument("--feature-set-json", default="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json")
    parser.add_argument("--tracking-uri", default=get_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Baselines")
    parser.add_argument("--output-root", default="experiments/outputs/tabular_baselines")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--holdout-split-csv", default=DEFAULT_HOLDOUT_SPLIT_CSV)
    parser.add_argument("--cv-split-csv", default=DEFAULT_CV_SPLIT_CSV)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument("--max-val-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional post-split row cap for smoke tests.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Runtime device for tabular estimators. Use `cuda` to enable GPU-capable backends.",
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
    ensure_expected_conda_env()
    global binary_classification_metrics
    global build_catboost_estimator
    global build_final_feature_frames
    global build_lightgbm_estimator
    global build_preprocessor
    global build_sklearn_estimator
    global build_torch_estimator
    global build_xgboost_estimator
    global device_uses_cuda
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

    validate_runtime_device(args.device)
    ensure_feature_set_json(args.eda_dir, args.feature_set_json)
    feature_payload = json.loads(Path(args.feature_set_json).read_text(encoding="utf-8"))
    raw_feature_sets = feature_payload.get("feature_sets", {})
    feature_payload["feature_sets"] = {
        normalize_feature_set_name(name): columns for name, columns in raw_feature_sets.items()
    }
    all_feature_sets = set(feature_payload["feature_sets"].keys())
    if not all_feature_sets:
        raise RuntimeError(f"No feature sets were found in {args.feature_set_json}")
    requested_feature_sets = normalize_feature_set_names(args.feature_sets)
    if requested_feature_sets:
        unknown_feature_sets = sorted(set(requested_feature_sets) - all_feature_sets)
        if unknown_feature_sets:
            raise RuntimeError(
                f"Requested feature sets are not available in {args.feature_set_json}: {unknown_feature_sets}"
            )
        available_feature_sets = set(requested_feature_sets)
    else:
        available_feature_sets = set(all_feature_sets)

    set_global_seed(args.seed)
    frame = load_merged_dataframe(args.dataset_root)
    target_column = feature_payload["target_column"]
    if target_column != DEFAULT_TARGET_COLUMN:
        raise RuntimeError(f"Unexpected target column in feature set JSON: {target_column}")
    split_definition = load_locked_split_definition(
        holdout_split_csv=args.holdout_split_csv,
        cv_split_csv=args.cv_split_csv,
        cv_fold=args.cv_fold,
    )
    sample_ids = frame["isic_id"].astype(str).copy()
    use_final_inputs = is_final_inputs_feature_payload(feature_payload)
    final_feature_frames = (
        build_final_feature_frames(frame, args.eda_dir, sorted(available_feature_sets))
        if use_final_inputs
        else {}
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    for spec in get_model_specs(args.models):
        parent_run_name = sanitize_run_name(spec.name)
        combinations = [
            combination
            for combination in expand_search_space(spec.search_space)
            if combination.get("feature_set") in available_feature_sets
        ]
        best_result = None
        best_run_name = None

        with mlflow.start_run(run_name=parent_run_name):
            mlflow.set_tags(
                {
                    "experiment_family": "isic2024_tabular_baseline",
                    "model_name": spec.name,
                    "role": "model_parent",
                    "dataset_name": "ISIC2024-challenge",
                }
            )
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "cv_fold": args.cv_fold,
                    "holdout_split_csv": str(Path(args.holdout_split_csv).resolve()),
                    "cv_split_csv": str(Path(args.cv_split_csv).resolve()),
                    "split_source": "locked_split_csv",
                    "max_train_rows": args.max_train_rows,
                    "max_val_rows": args.max_val_rows,
                    "max_test_rows": args.max_test_rows,
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "split_group_policy": "patient_id -> lesion_id -> isic_id",
                    "primary_metric_name": PRIMARY_PAUC_METRIC,
                    "threshold_source": "validation_f1",
                    "selected_feature_sets": ",".join(sorted(available_feature_sets)),
                    "runtime_device": args.device,
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
                summary = train_and_evaluate(
                    frame=split_frame,
                    sample_ids=sample_ids,
                    split_definition=split_definition,
                    target_column=target_column,
                    model_name=spec.name,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    device=args.device,
                    include_test=False,
                    max_train_rows=args.max_train_rows,
                    max_val_rows=args.max_val_rows,
                    max_test_rows=args.max_test_rows,
                )

                with mlflow.start_run(run_name=trial_run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "experiment_family": "isic2024_tabular_baseline",
                            "model_name": spec.name,
                            "role": "hyperparameter_trial",
                            "feature_set": feature_set_name,
                        }
                    )
                    mlflow.log_params({f"hp_{key}": normalize_param_value(value) for key, value in hyperparameters.items()})
                    mlflow.log_param("feature_count", len(features))
                    mlflow.log_param("feature_set", feature_set_name)
                    mlflow.log_param("runtime_device", args.device)
                    mlflow.log_params(missing_value_policy_params(summary["missing_value_policy"]))
                    mlflow.log_dict(summary["split_summary"], "split_summary.json")
                    mlflow.log_dict(summary["metrics"], "metrics.json")
                    mlflow.log_dict(summary["hyperparameters"], "hyperparameters.json")
                    for metric_group, metrics in summary["metrics"].items():
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(f"{metric_group}_{metric_name}", float(metric_value))
                    mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
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
            best_final_summary = train_and_evaluate(
                frame=best_split_frame,
                sample_ids=sample_ids,
                split_definition=split_definition,
                target_column=target_column,
                model_name=spec.name,
                hyperparameters=best_result["hyperparameters"],
                output_dir=final_output_dir,
                device=args.device,
                include_test=True,
                max_train_rows=args.max_train_rows,
                max_val_rows=args.max_val_rows,
                max_test_rows=args.max_test_rows,
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
            mlflow.log_param("selected_threshold", best_final_summary["selected_threshold"])
            mlflow.log_param("threshold_source", best_final_summary["threshold_source"])
            for metric_name, metric_value in best_final_summary["metrics"]["test"].items():
                mlflow.log_metric(f"best_{metric_name}", float(metric_value))


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


def load_merged_dataframe(dataset_root: str):
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
    include_test: bool,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, Any]:
    from isic2024_multimodal.features.tabular_missing import missing_value_policy_summary

    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(hyperparameters["seed"])
    set_global_seed(seed)

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
    estimator = build_estimator(
        model_name=model_name,
        hyperparameters=hyperparameters,
        X_train=X_train,
        y_train=y_train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        device=device,
    )
    estimator.fit(X_train, y_train)

    val_labels, val_probabilities = predict_probabilities(estimator, X_val, y_val)
    selected_threshold = select_threshold_by_f1(val_labels, val_probabilities)
    metrics = {
        "train": evaluate_predictions(estimator, X_train, y_train, threshold=selected_threshold),
        "val": evaluate_predictions(estimator, X_val, y_val, threshold=selected_threshold),
    }
    if include_test:
        metrics["test"] = evaluate_predictions(estimator, X_test, y_test, threshold=selected_threshold)
    duration_seconds = time.time() - start

    summary = {
        "model_name": model_name,
        "hyperparameters": {key: normalize_param_value(value) for key, value in hyperparameters.items()},
        "threshold_source": "validation_f1",
        "selected_threshold": selected_threshold,
        "split_source": "locked_split_csv",
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
            "holdout_split_csv": split_definition["holdout_split_csv"],
            "cv_split_csv": split_definition["cv_split_csv"],
            "cv_fold": split_definition["cv_fold"],
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "runtime_device": device,
            "max_train_rows": max_train_rows,
            "max_val_rows": max_val_rows,
            "max_test_rows": max_test_rows,
        },
        "metrics": metrics,
        "duration_seconds": duration_seconds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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


def load_locked_split_definition(*, holdout_split_csv: str, cv_split_csv: str, cv_fold: int) -> dict[str, Any]:
    import pandas as pd

    holdout_path = Path(holdout_split_csv)
    cv_path = Path(cv_split_csv)
    if not holdout_path.exists() or not cv_path.exists():
        missing_paths = [str(path) for path in [holdout_path, cv_path] if not path.exists()]
        raise FileNotFoundError(
            "Locked split CSV files are required for paper-valid tabular baselines. "
            f"Missing: {missing_paths}. Generate them with: "
            "`PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset`"
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
        train_frame = preprocessor.fit_transform(X_train)
        estimator = build_catboost_estimator(hyperparameters, device=device)
        estimator.fit(train_frame, y_train, cat_features=categorical_columns)
        return CatBoostWrapper(estimator=estimator, preprocessor=preprocessor, categorical_columns=categorical_columns)

    raise ValueError(f"Unsupported tabular model: {model_name}")


class CatBoostWrapper:
    def __init__(self, *, estimator, preprocessor, categorical_columns: list[str]) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.categorical_columns = categorical_columns

    def fit(self, X, y):
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


def validate_runtime_device(device: str) -> None:
    if not device_uses_cuda(device):
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
