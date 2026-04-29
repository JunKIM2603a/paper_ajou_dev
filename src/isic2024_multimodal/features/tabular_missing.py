from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CATEGORICAL_MISSING_VALUE = "__missing__"
NUMERIC_IMPUTATION_STRATEGY = "train_median"
CATEGORICAL_IMPUTATION_STRATEGY = "constant___missing__"
STRICT_NUMERIC_MISSING_INDICATOR_COLUMNS = ["age_approx"]
CATBOOST_CATEGORICAL_HANDLING = "native_cat_features_with___missing__"


def missing_value_policy_summary() -> dict[str, Any]:
    return {
        "numeric_imputation": NUMERIC_IMPUTATION_STRATEGY,
        "categorical_imputation": CATEGORICAL_IMPUTATION_STRATEGY,
        "categorical_fill_value": CATEGORICAL_MISSING_VALUE,
        "numeric_missing_indicators": list(STRICT_NUMERIC_MISSING_INDICATOR_COLUMNS),
        "catboost_categorical_handling": CATBOOST_CATEGORICAL_HANDLING,
        "fit_scope": "fold_train_only",
    }


def build_tabular_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    indicator_columns = [column for column in STRICT_NUMERIC_MISSING_INDICATOR_COLUMNS if column in numeric_columns]
    if indicator_columns:
        transformers.append(
            (
                "numeric_missing_indicator",
                FunctionTransformer(_missing_indicator_matrix, validate=False),
                indicator_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_MISSING_VALUE)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _missing_indicator_matrix(values):
    import pandas as pd

    return pd.DataFrame(values).isna().astype(np.float32).to_numpy()


@dataclass
class CatBoostMissingValuePreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_missing_indicator_columns: list[str] | None = None

    def __post_init__(self) -> None:
        if self.numeric_missing_indicator_columns is None:
            self.numeric_missing_indicator_columns = [
                column for column in STRICT_NUMERIC_MISSING_INDICATOR_COLUMNS if column in self.numeric_columns
            ]
        self.numeric_medians_: dict[str, float] = {}

    def fit(self, X, y=None):
        import pandas as pd

        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="coerce")
            median = values.median()
            self.numeric_medians_[column] = 0.0 if pd.isna(median) else float(median)
        return self

    def transform(self, X):
        import pandas as pd

        frame = X.copy()
        for column in self.numeric_columns:
            original_values = pd.to_numeric(frame[column], errors="coerce")
            if column in self.numeric_missing_indicator_columns:
                frame[f"{column}__missing"] = original_values.isna().astype(int)
            frame[column] = original_values.fillna(self.numeric_medians_[column])
        for column in self.categorical_columns:
            frame[column] = frame[column].fillna(CATEGORICAL_MISSING_VALUE).astype(str)
        return frame

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @property
    def numeric_missing_indicator_feature_names(self) -> list[str]:
        return [f"{column}__missing" for column in self.numeric_missing_indicator_columns or []]
