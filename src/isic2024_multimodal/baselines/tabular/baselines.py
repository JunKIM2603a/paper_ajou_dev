from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isic2024_multimodal.features.tabular_missing import build_tabular_preprocessor
from isic2024_multimodal.features.tabular_terms import STRICT_BASE, STRICT_FE, STRICT_MAIN_INPUT
from isic2024_multimodal.models.tabular.torch_estimator import TorchTabularEstimator


DEFAULT_STRICT_FEATURE_SETS = [STRICT_BASE, STRICT_FE, STRICT_MAIN_INPUT]


@dataclass(frozen=True)
class TabularModelSpec:
    name: str
    family: str
    search_space: dict[str, list[Any]]


MODEL_SPECS: dict[str, TabularModelSpec] = {
    "logistic_regression": TabularModelSpec(
        name="logistic_regression",
        family="sklearn",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["logistic_regression"],
            "C": [1.0],
            "max_iter": [1000],
        },
    ),
    "svm": TabularModelSpec(
        name="svm",
        family="sklearn",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["svm"],
            "C": [1.0],
            "max_iter": [20000],
        },
    ),
    "mlp": TabularModelSpec(
        name="mlp",
        family="sklearn",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["mlp"],
            "hidden_layer_sizes": [(64, 32)],
            "alpha": [0.0001],
            "max_iter": [50],
        },
    ),
    "xgboost": TabularModelSpec(
        name="xgboost",
        family="xgboost",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["xgboost"],
            "n_estimators": [200],
            "max_depth": [6],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    ),
    "catboost": TabularModelSpec(
        name="catboost",
        family="catboost",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["catboost"],
            "iterations": [300],
            "depth": [6],
            "learning_rate": [0.05],
        },
    ),
    "lightgbm": TabularModelSpec(
        name="lightgbm",
        family="lightgbm",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["lightgbm"],
            "n_estimators": [300],
            "num_leaves": [31],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    ),
    "ft_transformer": TabularModelSpec(
        name="ft_transformer",
        family="torch",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["ft_transformer"],
            "max_iter": [50],
            "learning_rate": [0.001],
            "weight_decay": [0.00001],
            "d_token": [64],
            "n_blocks": [2],
            "n_heads": [4],
            "attention_dropout": [0.1],
            "ffn_dropout": [0.1],
        },
    ),
    "ft_transformer_external": TabularModelSpec(
        name="ft_transformer_external",
        family="torch_external",
        search_space={
            "feature_set": DEFAULT_STRICT_FEATURE_SETS,
            "model_name": ["ft_transformer_external"],
            "max_iter": [50],
            "learning_rate": [0.001],
            "weight_decay": [0.00001],
            "d_token": [64],
            "n_blocks": [2],
            "n_heads": [4],
            "attention_dropout": [0.1],
            "ffn_dropout": [0.1],
        },
    ),
}


def get_model_specs(model_names: list[str] | None = None) -> list[TabularModelSpec]:
    if not model_names:
        return list(MODEL_SPECS.values())
    return [MODEL_SPECS[name] for name in model_names]


def split_feature_types(frame) -> tuple[list[str], list[str]]:
    numeric_columns = []
    categorical_columns = []
    for column in frame.columns:
        if np.issubdtype(frame[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    return build_tabular_preprocessor(numeric_columns, categorical_columns)


def build_sklearn_estimator(model_name: str, hyperparameters: dict[str, Any]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC

    model_hyperparameters = strip_common_hyperparameters(hyperparameters)
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=int(hyperparameters["seed"]),
            **model_hyperparameters,
        )
    elif model_name == "svm":
        estimator = LinearSVC(
            class_weight="balanced",
            dual=False,
            random_state=int(hyperparameters["seed"]),
            **model_hyperparameters,
        )
    elif model_name == "mlp":
        estimator = MLPClassifier(
            random_state=int(hyperparameters["seed"]),
            early_stopping=True,
            **model_hyperparameters,
        )
    else:
        raise ValueError(f"Unsupported sklearn baseline model: {model_name}")
    return estimator


def device_uses_cuda(device: str | None) -> bool:
    return bool(device) and str(device).startswith("cuda")


def build_xgboost_estimator(hyperparameters: dict[str, Any], scale_pos_weight: float, *, device: str = "cpu"):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for the xgboost tabular baseline.") from exc

    estimator_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(hyperparameters["seed"]),
        scale_pos_weight=scale_pos_weight,
        **strip_common_hyperparameters(hyperparameters),
    )
    if device_uses_cuda(device):
        estimator_kwargs.update(
            {
                "tree_method": "hist",
                "device": "cuda",
            }
        )
    else:
        estimator_kwargs["n_jobs"] = 1
    return XGBClassifier(**estimator_kwargs)


def build_catboost_estimator(hyperparameters: dict[str, Any], *, device: str = "cpu"):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost is required for the catboost tabular baseline.") from exc

    estimator_kwargs = dict(
        loss_function="Logloss",
        random_seed=int(hyperparameters["seed"]),
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
        **strip_common_hyperparameters(hyperparameters),
    )
    if device_uses_cuda(device):
        estimator_kwargs["task_type"] = "GPU"
        estimator_kwargs["eval_metric"] = "Logloss"
    else:
        estimator_kwargs["eval_metric"] = "PRAUC"
    return CatBoostClassifier(**estimator_kwargs)


def build_lightgbm_estimator(hyperparameters: dict[str, Any], scale_pos_weight: float, *, device: str = "cpu"):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required for the lightgbm tabular baseline.") from exc

    estimator_kwargs = dict(
        objective="binary",
        random_state=int(hyperparameters["seed"]),
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
        **strip_common_hyperparameters(hyperparameters),
    )
    if device_uses_cuda(device):
        estimator_kwargs["device_type"] = "gpu"
    else:
        estimator_kwargs["n_jobs"] = 1
    return LGBMClassifier(**estimator_kwargs)


def build_torch_estimator(
    model_name: str,
    hyperparameters: dict[str, Any],
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_pos_weight: float,
    device: str,
):
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    return TorchTabularEstimator(
        model_name=model_name,
        preprocessor=preprocessor,
        hyperparameters=hyperparameters,
        device=device,
        scale_pos_weight=scale_pos_weight,
    )


def strip_common_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    excluded = {"model_name", "feature_set", "seed"}
    return {key: value for key, value in hyperparameters.items() if key not in excluded}
