from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from isic2024_benchmark.config_utils import expand_search_space, sanitize_run_name
from isic2024_benchmark.reproducibility import DEFAULT_SEED, set_global_seed
from isic2024_benchmark.runtime_env import ensure_expected_conda_env
from isic2024_benchmark.split_utils import split_group_ids
from isic2024_benchmark.tabular_baselines import (
    build_catboost_estimator,
    build_preprocessor,
    build_sklearn_estimator,
    build_xgboost_estimator,
    get_model_specs,
    split_feature_types,
)
from isic2024_benchmark.tabular_data import DEFAULT_DATASET_ROOT, DEFAULT_TARGET_COLUMN, load_tabular_dataframe
from isic2024_benchmark.tabular_feature_sets import recommend_feature_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISIC2024 challenge tabular baseline models.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--eda-dir", default="artifacts/eda/isic2024")
    parser.add_argument("--feature-set-json", default="artifacts/eda/isic2024/feature_sets_recommended.json")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Benchmark")
    parser.add_argument("--output-root", default="artifacts/tabular_runs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["logistic_regression", "svm", "mlp", "xgboost", "catboost"],
    )
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "pandas and mlflow are required to run tabular baselines. Activate the conda env and install dependencies."
        ) from exc

    args = parse_args()
    ensure_feature_set_json(args.eda_dir, args.feature_set_json)
    feature_payload = json.loads(Path(args.feature_set_json).read_text(encoding="utf-8"))

    set_global_seed(args.seed)
    frame = load_merged_dataframe(args.dataset_root)
    target_column = feature_payload["target_column"]
    if target_column != DEFAULT_TARGET_COLUMN:
        raise RuntimeError(f"Unexpected target column in feature set JSON: {target_column}")
    group_ids = frame["split_group_id"].copy()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    for spec in get_model_specs(args.models):
        parent_run_name = sanitize_run_name(spec.name)
        combinations = expand_search_space(spec.search_space)
        best_result = None
        best_run_name = None

        with mlflow.start_run(run_name=parent_run_name):
            mlflow.set_tags(
                {
                    "benchmark": "ISIC2024-tabular",
                    "model_name": spec.name,
                    "role": "model_parent",
                    "dataset_name": "ISIC2024-challenge",
                }
            )
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "test_size": args.test_size,
                    "validation_size": args.validation_size,
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "split_group_policy": "patient_id -> lesion_id -> isic_id",
                }
            )

            for index, combination in enumerate(combinations, start=1):
                trial_seed = args.seed + index - 1
                hyperparameters = dict(combination)
                hyperparameters["seed"] = trial_seed
                feature_set_name = hyperparameters["feature_set"]
                features = feature_payload["feature_sets"][feature_set_name]
                split_frame = frame[features + [target_column]].copy()

                trial_run_name = f"{parent_run_name}_trial_{index:03d}"
                output_dir = Path(args.output_root) / parent_run_name / trial_run_name
                output_dir.mkdir(parents=True, exist_ok=True)
                summary = train_and_evaluate(
                    frame=split_frame,
                    group_ids=group_ids,
                    target_column=target_column,
                    model_name=spec.name,
                    hyperparameters=hyperparameters,
                    test_size=args.test_size,
                    validation_size=args.validation_size,
                    output_dir=output_dir,
                )

                with mlflow.start_run(run_name=trial_run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "benchmark": "ISIC2024-tabular",
                            "model_name": spec.name,
                            "role": "hyperparameter_trial",
                            "feature_set": feature_set_name,
                        }
                    )
                    mlflow.log_params({f"hp_{key}": normalize_param_value(value) for key, value in hyperparameters.items()})
                    mlflow.log_param("feature_count", len(features))
                    mlflow.log_param("feature_set", feature_set_name)
                    mlflow.log_dict(summary["split_summary"], "split_summary.json")
                    mlflow.log_dict(summary["metrics"], "metrics.json")
                    mlflow.log_dict(summary["hyperparameters"], "hyperparameters.json")
                    for metric_group, metrics in summary["metrics"].items():
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(f"{metric_group}_{metric_name}", float(metric_value))
                    mlflow.log_metric("duration_seconds", float(summary["duration_seconds"]))
                    mlflow.log_artifact(str(output_dir / "summary.json"))

                score = summary["metrics"]["test"]["average_precision"]
                if best_result is None or score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "summary": summary,
                        "hyperparameters": hyperparameters,
                    }
                    best_run_name = trial_run_name

            if best_result is None:
                raise RuntimeError(f"No successful tabular trials completed for {spec.name}")

            mlflow.log_dict(best_result["summary"], "best_summary.json")
            mlflow.set_tag("best_child_run_name", best_run_name)
            mlflow.log_params(
                {
                    f"best_hp_{key}": normalize_param_value(value)
                    for key, value in best_result["hyperparameters"].items()
                }
            )
            mlflow.log_param("best_feature_set", best_result["hyperparameters"]["feature_set"])
            for metric_name, metric_value in best_result["summary"]["metrics"]["test"].items():
                mlflow.log_metric(f"best_{metric_name}", float(metric_value))


def ensure_feature_set_json(eda_dir: str, feature_set_json: str) -> None:
    path = Path(feature_set_json)
    if path.exists():
        return
    payload = recommend_feature_sets(eda_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_merged_dataframe(dataset_root: str):
    frame = load_tabular_dataframe(dataset_root, include_image_columns=False)
    if DEFAULT_TARGET_COLUMN not in frame.columns:
        raise RuntimeError(f"Target column '{DEFAULT_TARGET_COLUMN}' not found in {dataset_root}")
    return frame


def train_and_evaluate(
    *,
    frame,
    group_ids,
    target_column: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    test_size: float,
    validation_size: float,
    output_dir: Path,
) -> dict[str, Any]:
    import pandas as pd

    start = time.time()
    seed = int(hyperparameters["seed"])
    set_global_seed(seed)

    X = frame.drop(columns=[target_column]).copy()
    y = frame[target_column].astype(float).astype(int)
    split_sets = build_split_sets(
        group_ids=group_ids,
        y=y,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    for column in X.columns:
        X[column] = X[column].where(X[column].notna(), None)

    train_mask = group_ids.isin(split_sets["train"])
    val_mask = group_ids.isin(split_sets["val"])
    test_mask = group_ids.isin(split_sets["test"])
    X_train = X.loc[train_mask].copy()
    X_val = X.loc[val_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_val = y.loc[val_mask].copy()
    y_test = y.loc[test_mask].copy()

    numeric_columns, categorical_columns = split_feature_types(X_train)
    estimator = build_estimator(
        model_name=model_name,
        hyperparameters=hyperparameters,
        X_train=X_train,
        y_train=y_train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    estimator.fit(X_train, y_train)

    metrics = {
        "train": evaluate_predictions(estimator, X_train, y_train),
        "val": evaluate_predictions(estimator, X_val, y_val),
        "test": evaluate_predictions(estimator, X_test, y_test),
    }
    duration_seconds = time.time() - start

    summary = {
        "model_name": model_name,
        "hyperparameters": {key: normalize_param_value(value) for key, value in hyperparameters.items()},
        "split_summary": {
            "num_train_rows": int(len(X_train)),
            "num_val_rows": int(len(X_val)),
            "num_test_rows": int(len(X_test)),
            "num_train_positive": int(y_train.sum()),
            "num_val_positive": int(y_val.sum()),
            "num_test_positive": int(y_test.sum()),
            "num_train_groups": int(len(split_sets["train"])),
            "num_val_groups": int(len(split_sets["val"])),
            "num_test_groups": int(len(split_sets["test"])),
            "split_group_policy": "patient_id -> lesion_id -> isic_id",
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        "metrics": metrics,
        "duration_seconds": duration_seconds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_split_sets(*, group_ids, y, validation_size: float, test_size: float, seed: int) -> dict[str, set[str]]:
    import pandas as pd

    group_frame = pd.DataFrame({"group_id": group_ids, "target": y})
    group_labels = group_frame.groupby("group_id")["target"].max().astype(int).to_dict()
    return split_group_ids(
        group_labels,
        validation_ratio=validation_size,
        test_ratio=test_size,
        seed=seed,
    )


def build_estimator(*, model_name: str, hyperparameters: dict[str, Any], X_train, y_train, numeric_columns, categorical_columns):
    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int(len(y_train) - positive_count), 1)
    scale_pos_weight = negative_count / positive_count

    if model_name in {"logistic_regression", "svm", "mlp"}:
        from sklearn.pipeline import Pipeline

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        estimator = build_sklearn_estimator(model_name, hyperparameters)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    if model_name == "xgboost":
        from sklearn.pipeline import Pipeline

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        estimator = build_xgboost_estimator(hyperparameters, scale_pos_weight=scale_pos_weight)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    if model_name == "catboost":
        import pandas as pd

        train_frame = X_train.copy()
        for column in categorical_columns:
            train_frame[column] = train_frame[column].fillna("__missing__").astype(str)
        for column in numeric_columns:
            train_frame[column] = pd.to_numeric(train_frame[column], errors="coerce")
        estimator = build_catboost_estimator(hyperparameters)
        estimator.fit(train_frame, y_train, cat_features=categorical_columns)
        return CatBoostWrapper(estimator=estimator, categorical_columns=categorical_columns, numeric_columns=numeric_columns)

    raise ValueError(f"Unsupported tabular model: {model_name}")


class CatBoostWrapper:
    def __init__(self, *, estimator, categorical_columns: list[str], numeric_columns: list[str]) -> None:
        self.estimator = estimator
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    def fit(self, X, y):
        return self

    def _prepare(self, X):
        import pandas as pd

        frame = X.copy()
        for column in self.categorical_columns:
            frame[column] = frame[column].fillna("__missing__").astype(str)
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def predict(self, X):
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._prepare(X))


def evaluate_predictions(estimator, X, y_true) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    else:
        y_score = y_pred

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0
    try:
        average_precision = average_precision_score(y_true, y_score)
    except ValueError:
        average_precision = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc),
        "average_precision": float(average_precision),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def normalize_param_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


if __name__ == "__main__":
    main()
