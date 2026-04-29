from __future__ import annotations

import sys
import types

import pandas as pd

from isic2024_multimodal.baselines.tabular.baselines import build_lightgbm_estimator
from isic2024_multimodal.cli.run_tabular_baseline import load_locked_split_definition, select_trial_score
from isic2024_multimodal.evaluation.metrics import select_threshold_by_f1, thresholded_binary_classification_metrics
from isic2024_multimodal.features.tabular_missing import (
    CATEGORICAL_MISSING_VALUE,
    CatBoostMissingValuePreprocessor,
    build_tabular_preprocessor,
    missing_value_policy_summary,
)


def write_split_csvs(tmp_path):
    holdout_path = tmp_path / "holdout.csv"
    cv_path = tmp_path / "cv.csv"
    holdout_frame = pd.DataFrame(
        [
            {"isic_id": "A", "patient_id": "P1", "lesion_id": "L1", "split": "train_validation_data"},
            {"isic_id": "B", "patient_id": "P2", "lesion_id": "L2", "split": "train_validation_data"},
            {"isic_id": "C", "patient_id": "P3", "lesion_id": "L3", "split": "train_validation_data"},
            {"isic_id": "D", "patient_id": "P4", "lesion_id": "L4", "split": "test_data"},
        ]
    )
    cv_frame = pd.DataFrame(
        [
            {"isic_id": "A", "patient_id": "P1", "lesion_id": "L1", "cv_validation_fold": 0},
            {"isic_id": "B", "patient_id": "P2", "lesion_id": "L2", "cv_validation_fold": 1},
            {"isic_id": "C", "patient_id": "P3", "lesion_id": "L3", "cv_validation_fold": 1},
        ]
    )
    holdout_frame.to_csv(holdout_path, index=False)
    cv_frame.to_csv(cv_path, index=False)
    return holdout_path, cv_path


def test_locked_split_definition_is_patient_disjoint(tmp_path) -> None:
    holdout_path, cv_path = write_split_csvs(tmp_path)

    split_definition = load_locked_split_definition(
        holdout_split_csv=str(holdout_path),
        cv_split_csv=str(cv_path),
        cv_fold=0,
    )

    assert split_definition["train_ids"] == {"B", "C"}
    assert split_definition["val_ids"] == {"A"}
    assert split_definition["test_ids"] == {"D"}
    assert split_definition["overlap_checks"] == {
        "train_val_patient_overlap": 0,
        "train_test_patient_overlap": 0,
        "val_test_patient_overlap": 0,
    }


def test_locked_split_does_not_depend_on_trial_seed(tmp_path) -> None:
    holdout_path, cv_path = write_split_csvs(tmp_path)

    first = load_locked_split_definition(holdout_split_csv=str(holdout_path), cv_split_csv=str(cv_path), cv_fold=0)
    second = load_locked_split_definition(holdout_split_csv=str(holdout_path), cv_split_csv=str(cv_path), cv_fold=0)

    assert first["train_ids"] == second["train_ids"]
    assert first["val_ids"] == second["val_ids"]
    assert first["test_ids"] == second["test_ids"]


def test_threshold_selection_uses_validation_probabilities() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.05, 0.4, 0.45, 0.9]

    threshold = select_threshold_by_f1(labels, probabilities)
    metrics = thresholded_binary_classification_metrics(labels, probabilities, threshold=threshold)

    assert threshold == 0.45
    assert metrics["f1_score"] == 1.0
    assert metrics["threshold"] == threshold


def test_trial_selection_score_never_falls_back_to_test_metrics() -> None:
    summary = {
        "metrics": {
            "val": {
                "pauc_above_tpr80": float("nan"),
                "auc_roc": float("nan"),
                "average_precision": 0.25,
            },
            "test": {
                "average_precision": 0.99,
            },
        }
    }

    assert select_trial_score(summary) == 0.25


def test_lightgbm_builder_receives_train_only_scale_pos_weight(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_module = types.SimpleNamespace(LGBMClassifier=FakeLGBMClassifier)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_module)

    build_lightgbm_estimator(
        {
            "seed": 123,
            "model_name": "lightgbm",
            "feature_set": "strict_main_input",
            "n_estimators": 10,
        },
        scale_pos_weight=17.5,
        device="cpu",
    )

    assert captured_kwargs["scale_pos_weight"] == 17.5
    assert captured_kwargs["random_state"] == 123
    assert captured_kwargs["n_estimators"] == 10
    assert "model_name" not in captured_kwargs
    assert "feature_set" not in captured_kwargs


def test_tabular_preprocessor_uses_train_median_and_missing_category() -> None:
    train = pd.DataFrame(
        {
            "age_approx": [10.0, None, 30.0],
            "sex": ["male", None, "female"],
            "anatom_site_general": [None, "torso", "lower extremity"],
        }
    )
    val = pd.DataFrame(
        {
            "age_approx": [None],
            "sex": [None],
            "anatom_site_general": ["head/neck"],
        }
    )

    preprocessor = build_tabular_preprocessor(
        numeric_columns=["age_approx"],
        categorical_columns=["sex", "anatom_site_general"],
    )
    train_matrix = preprocessor.fit_transform(train)
    val_matrix = preprocessor.transform(val)

    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    categorical_imputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]

    assert numeric_imputer.statistics_.tolist() == [20.0]
    assert categorical_imputer.statistics_.tolist() == [CATEGORICAL_MISSING_VALUE, CATEGORICAL_MISSING_VALUE]
    assert train_matrix[1, 1] == 1.0
    assert val_matrix[0, 1] == 1.0


def test_catboost_missing_preprocessor_preserves_native_categoricals() -> None:
    train = pd.DataFrame(
        {
            "age_approx": [20.0, None, 60.0],
            "tbp_lv_A": [1.0, None, 3.0],
            "sex": ["male", None, "female"],
        }
    )
    val = pd.DataFrame({"age_approx": [None], "tbp_lv_A": [None], "sex": [None]})
    preprocessor = CatBoostMissingValuePreprocessor(
        numeric_columns=["age_approx", "tbp_lv_A"],
        categorical_columns=["sex"],
    )

    train_prepared = preprocessor.fit_transform(train)
    val_prepared = preprocessor.transform(val)

    assert preprocessor.numeric_medians_ == {"age_approx": 40.0, "tbp_lv_A": 2.0}
    assert train_prepared.loc[1, "age_approx"] == 40.0
    assert train_prepared.loc[1, "age_approx__missing"] == 1
    assert val_prepared.loc[0, "age_approx"] == 40.0
    assert val_prepared.loc[0, "age_approx__missing"] == 1
    assert val_prepared.loc[0, "sex"] == CATEGORICAL_MISSING_VALUE
    assert "sex__missing" not in val_prepared.columns


def test_missing_value_policy_is_logged_as_explicit_contract() -> None:
    policy = missing_value_policy_summary()

    assert policy["numeric_imputation"] == "train_median"
    assert policy["categorical_imputation"] == "constant___missing__"
    assert policy["numeric_missing_indicators"] == ["age_approx"]
    assert policy["catboost_categorical_handling"] == "native_cat_features_with___missing__"


def test_repo_native_ft_transformer_forward_shape() -> None:
    import torch

    from isic2024_multimodal.models.tabular.ft_transformer import FTTransformerBinaryModel

    model = FTTransformerBinaryModel(input_dim=6, d_token=8, n_blocks=1, n_heads=2)
    logits = model(torch.randn(4, 6))

    assert tuple(logits.shape) == (4,)
