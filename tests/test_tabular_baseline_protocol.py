from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from isic2024_multimodal.baselines.tabular.baselines import build_lightgbm_estimator
from isic2024_multimodal.cli.run_all_tabular_models import (
    FoldSelection,
    build_command,
    build_job_plan,
    build_preflight_command,
    resolve_nested_fold_selections,
)
from isic2024_multimodal.cli.run_tabular_baseline import (
    load_locked_split_definition,
    load_nested_split_definition,
    run_preflight,
    select_trial_score,
    validate_runtime_device,
)
from isic2024_multimodal.evaluation.metrics import (
    partial_auc_above_min_tpr,
    select_threshold_by_f1,
    thresholded_binary_classification_metrics,
)
from isic2024_multimodal.features.tabular_missing import (
    CATEGORICAL_MISSING_VALUE,
    CatBoostMissingValuePreprocessor,
    build_tabular_preprocessor,
    missing_value_policy_summary,
)
from isic2024_multimodal.reporting.mlflow_report import build_filter_string
from isic2024_multimodal.utils import device as device_utils
from isic2024_multimodal.utils.progress import (
    estimate_remaining_seconds,
    format_eta,
    format_progress_duration,
    progress_index_label,
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


def write_nested_split_csv(tmp_path):
    nested_path = tmp_path / "nested.csv"
    nested_frame = pd.DataFrame(
        [
            {"isic_id": "A", "patient_id": "P1", "lesion_id": "L1", "outer_fold": 0, "cv_test_fold": 0, "inner_fold": 0, "split_role": "outer_test"},
            {"isic_id": "B", "patient_id": "P2", "lesion_id": "L2", "outer_fold": 0, "cv_test_fold": 0, "inner_fold": 0, "split_role": "inner_validation"},
            {"isic_id": "C", "patient_id": "P3", "lesion_id": "L3", "outer_fold": 0, "cv_test_fold": 0, "inner_fold": 0, "split_role": "inner_train"},
            {"isic_id": "D", "patient_id": "P4", "lesion_id": "L4", "outer_fold": 0, "cv_test_fold": 0, "inner_fold": 0, "split_role": "inner_train"},
        ]
    )
    nested_frame.to_csv(nested_path, index=False)
    return nested_path


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


def test_nested_split_definition_is_patient_disjoint(tmp_path) -> None:
    nested_path = write_nested_split_csv(tmp_path)

    split_definition = load_nested_split_definition(
        nested_split_csv=str(nested_path),
        outer_fold=0,
        inner_fold=0,
    )

    assert split_definition["train_ids"] == {"C", "D"}
    assert split_definition["val_ids"] == {"B"}
    assert split_definition["test_ids"] == {"A"}
    assert split_definition["split_protocol"] == "nested_cv"
    assert split_definition["overlap_checks"] == {
        "inner_train_inner_validation_patient_overlap": 0,
        "inner_train_outer_test_patient_overlap": 0,
        "inner_validation_outer_test_patient_overlap": 0,
    }


def test_threshold_selection_uses_validation_probabilities() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.05, 0.4, 0.45, 0.9]

    threshold = select_threshold_by_f1(labels, probabilities)
    metrics = thresholded_binary_classification_metrics(labels, probabilities, threshold=threshold)

    assert threshold == 0.45
    assert metrics["f1_score"] == 1.0
    assert metrics["threshold"] == threshold


def test_threshold_selection_matches_bruteforce_with_ties() -> None:
    labels = [0, 1, 0, 1, 0, 1, 0]
    probabilities = [0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.95]
    candidate_thresholds = sorted(set(probabilities))
    best_threshold = candidate_thresholds[0]
    best_score = -1.0
    for threshold in candidate_thresholds:
        metrics = thresholded_binary_classification_metrics(labels, probabilities, threshold=threshold)
        score = metrics["f1_score"]
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = threshold

    assert select_threshold_by_f1(labels, probabilities) == best_threshold


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
        device="cuda",
    )

    assert captured_kwargs["scale_pos_weight"] == 17.5
    assert captured_kwargs["random_state"] == 123
    assert captured_kwargs["n_estimators"] == 10
    assert captured_kwargs["n_jobs"] == 1
    assert "device_type" not in captured_kwargs
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


def test_torch_ft_estimator_uses_small_default_batches() -> None:
    from isic2024_multimodal.models.tabular.torch_estimator import TorchTabularEstimator

    estimator = TorchTabularEstimator(
        model_name="ft_transformer",
        preprocessor=build_tabular_preprocessor(numeric_columns=["age_approx"], categorical_columns=["sex"]),
        hyperparameters={"seed": 42},
        device="cpu",
        scale_pos_weight=1.0,
    )

    assert estimator.runtime_params()["train_batch_size"] == 2048
    assert estimator.runtime_params()["predict_batch_size"] == 2048


def test_torch_ft_estimator_mini_batch_fit_and_predict() -> None:
    from isic2024_multimodal.models.tabular.torch_estimator import TorchTabularEstimator

    X = pd.DataFrame(
        {
            "age_approx": [20.0, 30.0, None, 50.0, 60.0, 70.0, 40.0, 35.0],
            "sex": ["male", "female", None, "male", "female", "male", "female", None],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 0, 0, 1])
    estimator = TorchTabularEstimator(
        model_name="ft_transformer",
        preprocessor=build_tabular_preprocessor(numeric_columns=["age_approx"], categorical_columns=["sex"]),
        hyperparameters={
            "seed": 42,
            "max_iter": 1,
            "d_token": 8,
            "n_blocks": 1,
            "n_heads": 2,
            "batch_size": 4,
            "predict_batch_size": 3,
        },
        device="cpu",
        scale_pos_weight=1.0,
    )

    estimator.fit(X, y)
    probabilities = estimator.predict_proba(X)

    assert probabilities.shape == (8, 2)
    assert estimator.runtime_params()["train_batch_size"] == 4
    assert estimator.runtime_params()["predict_batch_size"] == 3


def test_run_all_tabular_gpu_command_passes_cuda_device() -> None:
    args = types.SimpleNamespace(
        dataset_root="data/raw",
        eda_dir="experiments/evidence/eda/isic_2024",
        feature_set_json="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json",
        experiment_name="test",
        run_group_id="unit_group",
        output_root="experiments/outputs/tabular_baselines_smoke",
        tracking_uri="file:experiments/logs/mlruns",
        seed=42,
        split_seed=42,
        split_protocol="nested_cv",
        nested_split_csv="data/splits/isic2024_official_train_nested_5x4_seed42.csv",
        outer_fold=0,
        inner_fold=0,
        cv_fold=0,
        holdout_split_csv="data/splits/isic2024_train_validation_test_split_seed42.csv",
        cv_split_csv="data/splits/isic2024_train_validation_5fold_seed42.csv",
        feature_sets=["strict_main_input"],
        models=["xgboost", "lightgbm"],
        max_train_rows=1000,
        max_val_rows=500,
        max_test_rows=500,
    )

    command = build_command("xgboost", args, device=0)
    preflight_command = build_preflight_command(args, device=0)

    assert command[command.index("--device") + 1] == "cuda"
    assert preflight_command[preflight_command.index("--device") + 1] == "cuda"
    assert command[command.index("--run-group-id") + 1] == "unit_group"
    assert preflight_command[preflight_command.index("--run-group-id") + 1] == "unit_group"
    assert "--preflight-only" in preflight_command


def test_device_resolver_auto_prefers_cuda_when_usable(monkeypatch) -> None:
    monkeypatch.setattr(
        device_utils,
        "_cuda_status",
        lambda: {"available": True, "count": 2, "usable": True, "reason": None},
    )

    resolution = device_utils.resolve_device("auto")
    list_resolution = device_utils.resolve_device_list(None)

    assert resolution.requested_device == "auto"
    assert resolution.resolved_device == "cuda"
    assert resolution.fallback_reason is None
    assert list_resolution.resolved_devices == [0]
    assert list_resolution.fallback_reason is None


def test_device_resolver_falls_back_to_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        device_utils,
        "_cuda_status",
        lambda: {
            "available": False,
            "count": 0,
            "usable": False,
            "reason": "torch.cuda.is_available() is False",
        },
    )

    resolution = device_utils.resolve_device("cuda")
    list_resolution = device_utils.resolve_device_list([0])

    assert resolution.requested_device == "cuda"
    assert resolution.resolved_device == "cpu"
    assert "torch.cuda.is_available" in str(resolution.fallback_reason)
    assert list_resolution.requested_devices == [0]
    assert list_resolution.resolved_devices == []
    assert "torch.cuda.is_available" in str(list_resolution.fallback_reason)


def test_run_all_tabular_cpu_policy_passes_cpu_device() -> None:
    args = types.SimpleNamespace(
        dataset_root="data/raw",
        eda_dir="experiments/evidence/eda/isic_2024",
        feature_set_json="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json",
        experiment_name="test",
        run_group_id="unit_group",
        output_root="experiments/outputs/tabular_baselines_smoke",
        tracking_uri="file:experiments/logs/mlruns",
        seed=42,
        split_seed=42,
        split_protocol="nested_cv",
        nested_split_csv="data/splits/isic2024_official_train_nested_5x4_seed42.csv",
        outer_fold=0,
        inner_fold=0,
        cv_fold=0,
        holdout_split_csv="data/splits/isic2024_train_validation_test_split_seed42.csv",
        cv_split_csv="data/splits/isic2024_train_validation_5fold_seed42.csv",
        feature_sets=["strict_main_input"],
        models=["xgboost"],
        max_train_rows=None,
        max_val_rows=None,
        max_test_rows=None,
        device_policy="cpu",
        devices=None,
        resolved_devices=[],
    )

    command = build_command("xgboost", args, device=None)
    preflight_command = build_preflight_command(args, device=None)

    assert command[command.index("--device") + 1] == "cpu"
    assert preflight_command[preflight_command.index("--device") + 1] == "cpu"


def test_run_all_tabular_discovers_all_nested_fold_pairs(tmp_path) -> None:
    nested_path = tmp_path / "nested.csv"
    pd.DataFrame(
        [
            {"isic_id": "A", "outer_fold": 1, "inner_fold": 1},
            {"isic_id": "B", "outer_fold": 0, "inner_fold": 1},
            {"isic_id": "C", "outer_fold": 0, "inner_fold": 0},
            {"isic_id": "D", "outer_fold": 1, "inner_fold": 0},
            {"isic_id": "E", "outer_fold": 1, "inner_fold": 0},
        ]
    ).to_csv(nested_path, index=False)

    folds = resolve_nested_fold_selections(nested_path)

    assert [fold.label for fold in folds] == [
        "outer_00_inner_00",
        "outer_00_inner_01",
        "outer_01_inner_00",
        "outer_01_inner_01",
    ]


def test_progress_helpers_format_eta_and_index_labels() -> None:
    assert format_progress_duration(None) == "unknown"
    assert format_progress_duration(65) == "1m 5s"
    assert progress_index_label(3, 20) == "3/20"
    assert estimate_remaining_seconds(elapsed_seconds=100, completed_count=4, total_count=10) == 150
    assert format_eta(elapsed_seconds=100, completed_count=4, total_count=10) == "2m 30s"


def test_run_all_tabular_job_plan_tracks_model_fold_and_job_indices() -> None:
    folds = [
        FoldSelection(outer_fold=0, inner_fold=0),
        FoldSelection(outer_fold=0, inner_fold=1),
    ]

    plan = build_job_plan(["xgboost", "catboost"], fold_selections=folds)

    assert [item.job_index for item in plan] == [1, 2, 3, 4]
    assert [item.total_jobs for item in plan] == [4, 4, 4, 4]
    assert [item.model_index for item in plan] == [1, 2, 1, 2]
    assert [item.total_models for item in plan] == [2, 2, 2, 2]
    assert [item.fold_index for item in plan] == [1, 1, 2, 2]
    assert [item.total_folds for item in plan] == [2, 2, 2, 2]


def test_run_all_tabular_all_folds_command_scopes_fold_output(tmp_path) -> None:
    args = types.SimpleNamespace(
        dataset_root="data/raw",
        eda_dir="experiments/evidence/eda/isic_2024",
        feature_set_json="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json",
        experiment_name="test",
        run_group_id="unit_group",
        output_root=str(tmp_path / "tabular_outputs"),
        tracking_uri="file:experiments/logs/mlruns",
        seed=42,
        split_seed=42,
        split_protocol="nested_cv",
        nested_split_csv="data/splits/isic2024_official_train_nested_5x4_seed42.csv",
        outer_fold=0,
        inner_fold=0,
        cv_fold=0,
        holdout_split_csv="data/splits/isic2024_train_validation_test_split_seed42.csv",
        cv_split_csv="data/splits/isic2024_train_validation_5fold_seed42.csv",
        feature_sets=["strict_main_input"],
        models=["xgboost"],
        max_train_rows=None,
        max_val_rows=None,
        max_test_rows=None,
        all_folds=True,
    )
    fold = FoldSelection(outer_fold=3, inner_fold=2)

    command = build_command("xgboost", args, device=None, fold=fold)
    preflight_command = build_preflight_command(args, device=None, fold=fold)

    assert command[command.index("--outer-fold") + 1] == "3"
    assert command[command.index("--inner-fold") + 1] == "2"
    assert command[command.index("--output-root") + 1].endswith("tabular_outputs/outer_03_inner_02")
    assert preflight_command[preflight_command.index("--outer-fold") + 1] == "3"
    assert preflight_command[preflight_command.index("--inner-fold") + 1] == "2"
    assert preflight_command[preflight_command.index("--output-root") + 1].endswith(
        "tabular_outputs/outer_03_inner_02"
    )


def test_mlflow_report_filter_can_scope_run_group() -> None:
    assert build_filter_string("model_parent", "unit_group") == (
        "tags.role = 'model_parent' and tags.run_group_id = 'unit_group'"
    )


def test_isic_pauc_matches_official_raw_rescale_golden_value() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.01, 0.02, 0.20, 0.80, 0.30, 0.40, 0.90, 0.95]

    assert partial_auc_above_min_tpr(labels, probabilities, min_tpr=0.80) == pytest.approx(0.15)


def test_preflight_reports_missing_dataset(tmp_path) -> None:
    feature_set_json = tmp_path / "feature_sets.json"
    feature_set_json.write_text(
        '{"target_column": "target", "feature_sets": {"strict_main_input": ["age_approx"]}}',
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        dataset_root=str(tmp_path / "missing_raw"),
        eda_dir=str(tmp_path),
        feature_set_json=str(feature_set_json),
        output_root=str(tmp_path / "outputs"),
        holdout_split_csv=str(tmp_path / "holdout.csv"),
        cv_split_csv=str(tmp_path / "cv.csv"),
        cv_fold=0,
        models=["xgboost"],
        feature_sets=["strict_main_input"],
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="ISIC2024 dataset not found"):
        run_preflight(args)


def test_preflight_reports_missing_locked_splits(tmp_path) -> None:
    dataset_root = tmp_path / "raw"
    dataset_root.mkdir()
    pd.DataFrame(
        {
            "isic_id": ["A"],
            "patient_id": ["P1"],
            "target": [0],
            "age_approx": [40.0],
        }
    ).to_csv(dataset_root / "train-metadata.csv", index=False)
    feature_set_json = tmp_path / "feature_sets.json"
    feature_set_json.write_text(
        '{"target_column": "target", "feature_sets": {"strict_main_input": ["age_approx"]}}',
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        dataset_root=str(dataset_root),
        eda_dir=str(tmp_path),
        feature_set_json=str(feature_set_json),
        output_root=str(tmp_path / "outputs"),
        holdout_split_csv=str(tmp_path / "holdout.csv"),
        cv_split_csv=str(tmp_path / "cv.csv"),
        cv_fold=0,
        models=["xgboost"],
        feature_sets=["strict_main_input"],
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="Locked split CSV files are required"):
        run_preflight(args)


def test_cuda_runtime_validation_fails_when_torch_cuda_unavailable(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        validate_runtime_device("cuda")
