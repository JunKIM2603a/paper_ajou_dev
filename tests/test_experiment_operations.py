from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from isic2024_multimodal.cli.run_image_baseline import (
    _cleanup_torch_state,
    patient_balanced_sample_weights,
    select_trial_score,
)
from isic2024_multimodal.cli.run_all_image_models import build_command as build_image_command
from isic2024_multimodal.cli.run_baseline_suite import build_suite_commands
from isic2024_multimodal.cli.image_baseline_status import (
    artifact_status_for_model,
    build_status_records,
    classify_model_status,
    checkpoint_status_for_model,
)
from isic2024_multimodal.cli.run_experiment_family import build_tabular_command
from isic2024_multimodal.data.image_dataset import create_splits_from_locked_csvs
from isic2024_multimodal.experiments.dataset_specs import load_dataset_spec
from isic2024_multimodal.experiments.families import reset_family_outputs, resolve_family_paths
from isic2024_multimodal.experiments.nested_cv_summary import (
    collect_nested_cv_summary_records,
    select_outer_fold_records,
    summarize_selected_test_metrics,
    write_nested_cv_summary_outputs,
)
from isic2024_multimodal.experiments.registry import read_selection_registry, write_family_selection
from isic2024_multimodal.models.image.checkpoint_preflight import preflight_image_model_config
from isic2024_multimodal.reporting.mlflow_report import build_filter_string
from isic2024_multimodal.training.trainer import (
    build_image_binary_loss_config,
    evaluate_outputs,
    format_duration,
    positive_class_probabilities,
    train_only_pos_weight,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dataset_spec_rejects_privileged_ordinary_feature(tmp_path) -> None:
    feature_set_json = tmp_path / "features.json"
    write_json(
        feature_set_json,
        {
            "target_column": "target",
            "feature_sets": {"strict_main_input": ["age_approx", "iddx_full"]},
        },
    )
    spec_path = tmp_path / "dataset_spec.json"
    write_json(
        spec_path,
        {
            "dataset_id": "bad_spec",
            "dataset_root": "data/raw",
            "feature_set_json": str(feature_set_json),
            "feature_sets": ["strict_main_input"],
        },
    )

    with pytest.raises(ValueError, match="privileged"):
        load_dataset_spec(spec_path, repo_root=tmp_path)


def test_family_reset_only_removes_selected_family_paths(tmp_path) -> None:
    tabular_paths = resolve_family_paths(
        family="tabular_baselines",
        run_group_id="unit_run",
        repo_root=tmp_path,
    )
    image_paths = resolve_family_paths(
        family="image_baselines",
        run_group_id="unit_run",
        repo_root=tmp_path,
    )
    tabular_paths.output_root.mkdir(parents=True)
    tabular_paths.table_root.mkdir(parents=True)
    image_paths.output_root.mkdir(parents=True)
    image_paths.table_root.mkdir(parents=True)

    reset_family_outputs(tabular_paths)

    assert not tabular_paths.output_root.exists()
    assert not tabular_paths.table_root.exists()
    assert image_paths.output_root.exists()
    assert image_paths.table_root.exists()


def test_mlflow_filter_scopes_family_run_group_and_dataset() -> None:
    assert build_filter_string(
        "model_parent",
        "run_a",
        dataset_id="strict_main_input_v1",
        model_family="tabular_baselines",
    ) == (
        "tags.role = 'model_parent' and tags.run_group_id = 'run_a' "
        "and tags.dataset_id = 'strict_main_input_v1' and tags.model_family = 'tabular_baselines'"
    )


def test_registry_writes_best_selection_from_local_summaries(tmp_path) -> None:
    output_root = tmp_path / "experiments" / "outputs" / "tabular_baselines" / "run_a"
    first = output_root / "xgboost" / "best_final_test" / "summary.json"
    second = output_root / "catboost" / "best_final_test" / "summary.json"
    write_json(
        first,
        {
            "model_name": "xgboost",
            "selected_threshold": 0.4,
            "metrics": {"val": {"pauc_above_tpr80": 0.1}},
            "duration_seconds": 1.0,
        },
    )
    write_json(
        second,
        {
            "model_name": "catboost",
            "selected_threshold": 0.5,
            "metrics": {"val": {"pauc_above_tpr80": 0.2}},
            "duration_seconds": 2.0,
        },
    )

    selection = write_family_selection(
        family="tabular_baselines",
        run_group_id="run_a",
        dataset_id="strict_main_input_v1",
        output_root=output_root,
        table_root=tmp_path / "experiments" / "tables" / "tabular_baselines" / "run_a",
        config_path=tmp_path / "suite.json",
        dataset_spec_path=tmp_path / "dataset_spec.json",
        repo_root=tmp_path,
    )
    registry = read_selection_registry(
        tmp_path / "experiments" / "registry" / "selections" / "best_tabular_by_run_group.json"
    )

    assert selection is not None
    assert selection["model_name"] == "catboost"
    assert registry["run_a"]["validation_metric"] == 0.2


def test_nested_cv_summary_selects_outer_fold_candidates_by_validation_only(tmp_path) -> None:
    output_root = tmp_path / "experiments" / "outputs" / "tabular_baselines" / "run_a"
    write_json(
        output_root / "outer_00_inner_00" / "high_test" / "best_final_test" / "summary.json",
        {
            "model_name": "high_test",
            "outer_fold": 0,
            "inner_fold": 0,
            "selected_threshold": 0.4,
            "threshold_source": "inner_validation_f1",
            "hyperparameters": {"feature_set": "strict_main_input"},
            "metrics": {
                "val": {"pauc_above_tpr80": 0.10},
                "test": {
                    "pauc_above_tpr80": 0.90,
                    "auc_roc": 0.80,
                    "f1_score": 0.30,
                    "precision": 0.25,
                    "recall": 0.50,
                    "balanced_accuracy": 0.60,
                },
            },
        },
    )
    write_json(
        output_root / "outer_00_inner_00" / "high_test" / "high_test_trial_001" / "summary.json",
        {
            "model_name": "trial_duplicate_should_be_ignored",
            "outer_fold": 0,
            "inner_fold": 0,
            "metrics": {
                "val": {"pauc_above_tpr80": 0.99},
                "test": {"pauc_above_tpr80": 0.99},
            },
        },
    )
    write_json(
        output_root / "outer_00_inner_01" / "validation_winner" / "best_final_test" / "summary.json",
        {
            "model_name": "validation_winner",
            "outer_fold": 0,
            "inner_fold": 1,
            "selected_threshold": 0.5,
            "threshold_source": "inner_validation_f1",
            "hyperparameters": {"feature_set": "strict_main_input"},
            "metrics": {
                "val": {"pauc_above_tpr80": 0.20},
                "test": {
                    "pauc_above_tpr80": 0.10,
                    "auc_roc": 0.70,
                    "f1_score": 0.20,
                    "precision": 0.15,
                    "recall": 0.40,
                    "balanced_accuracy": 0.55,
                },
            },
        },
    )
    write_json(
        output_root / "outer_01_inner_00" / "fold_one" / "best_final_test" / "summary.json",
        {
            "model_name": "fold_one",
            "outer_fold": 1,
            "inner_fold": 0,
            "selected_threshold": 0.6,
            "threshold_source": "inner_validation_f1",
            "hyperparameters": {"feature_set": "strict_main_input"},
            "metrics": {
                "val": {"pauc_above_tpr80": 0.15},
                "test": {
                    "pauc_above_tpr80": 0.40,
                    "auc_roc": 0.75,
                    "f1_score": 0.25,
                    "precision": 0.20,
                    "recall": 0.45,
                    "balanced_accuracy": 0.58,
                },
            },
        },
    )
    write_json(
        output_root / "outer_01_inner_01" / "auc_only_high_value" / "best_final_test" / "summary.json",
        {
            "model_name": "auc_only_high_value",
            "outer_fold": 1,
            "inner_fold": 1,
            "selected_threshold": 0.7,
            "threshold_source": "inner_validation_f1",
            "hyperparameters": {"feature_set": "strict_main_input"},
            "metrics": {
                "val": {"auc_roc": 0.99},
                "test": {
                    "pauc_above_tpr80": 0.99,
                    "auc_roc": 0.99,
                    "f1_score": 0.90,
                    "precision": 0.90,
                    "recall": 0.90,
                    "balanced_accuracy": 0.90,
                },
            },
        },
    )

    records = collect_nested_cv_summary_records(
        output_root=output_root,
        family="tabular_baselines",
        run_group_id="run_a",
    )
    selected = select_outer_fold_records(records)
    selected_by_outer = {record.outer_fold: record for record in selected}
    metric_summary = summarize_selected_test_metrics(selected)
    pauc_summary = next(row for row in metric_summary if row["metric"] == "pauc_above_tpr80")
    table_root = tmp_path / "experiments" / "tables" / "tabular_baselines" / "run_a" / "nested_cv"

    manifest = write_nested_cv_summary_outputs(
        records=records,
        table_root=table_root,
        family="tabular_baselines",
        run_group_id="run_a",
        expected_outer_folds=2,
    )

    assert len(records) == 4
    assert selected_by_outer[0].model_name == "validation_winner"
    assert selected_by_outer[0].test_metrics["pauc_above_tpr80"] == 0.10
    assert selected_by_outer[1].model_name == "fold_one"
    assert pauc_summary["mean"] == pytest.approx(0.25)
    assert manifest["selected_outer_fold_count"] == 2
    assert (table_root / "nested_cv_all_candidates.csv").exists()
    assert (table_root / "nested_cv_outer_selection.csv").exists()
    assert (table_root / "nested_cv_metric_summary.csv").exists()
    markdown = (table_root / "nested_cv_summary.md").read_text(encoding="utf-8")
    assert "Selection uses validation metrics only" in markdown
    assert "validation_winner" in markdown


def test_nested_cv_summary_parses_image_style_summaries(tmp_path) -> None:
    output_root = tmp_path / "experiments" / "outputs" / "image_baselines" / "run_a"
    write_json(
        output_root / "resnet50" / "trial_001" / "summary.json",
        {
            "model_name": "resnet50",
            "split_summary": {"outer_fold": 2, "inner_fold": 3},
            "selected_threshold": 0.35,
            "threshold_source": "inner_validation_f1",
            "best_validation_metrics": {"auc_roc": 0.72, "average_precision": 0.08},
            "test_metrics": {
                "pauc_above_tpr80": 0.12,
                "auc_roc": 0.74,
                "f1_score": 0.22,
            },
        },
    )

    records = collect_nested_cv_summary_records(
        output_root=output_root,
        family="image_baselines",
        run_group_id="run_a",
    )

    assert len(records) == 1
    assert records[0].outer_fold == 2
    assert records[0].inner_fold == 3
    assert records[0].validation_metric_name == "auc_roc"
    assert records[0].validation_metric == 0.72
    assert records[0].test_metrics["pauc_above_tpr80"] == 0.12


def test_locked_image_split_uses_csv_membership(tmp_path) -> None:
    holdout = tmp_path / "holdout.csv"
    holdout.write_text(
        "isic_id,patient_id,split\n"
        "A,P1,train_validation_data\n"
        "B,P2,train_validation_data\n"
        "C,P3,test_data\n",
        encoding="utf-8",
    )
    cv = tmp_path / "cv.csv"
    cv.write_text(
        "isic_id,patient_id,cv_validation_fold\n"
        "A,P1,0\n"
        "B,P2,1\n",
        encoding="utf-8",
    )
    manifest = [
        {"image_path": "a.jpg", "label": 0, "group_id": "P1", "isic_id": "A", "metadata": {}},
        {"image_path": "b.jpg", "label": 1, "group_id": "P2", "isic_id": "B", "metadata": {}},
        {"image_path": "c.jpg", "label": 0, "group_id": "P3", "isic_id": "C", "metadata": {}},
    ]

    splits = create_splits_from_locked_csvs(
        manifest,
        holdout_split_csv=holdout,
        cv_split_csv=cv,
        cv_fold=0,
    )

    assert [sample.isic_id for sample in splits["val"]] == ["A"]
    assert [sample.isic_id for sample in splits["train"]] == ["B"]
    assert [sample.isic_id for sample in splits["test"]] == ["C"]


def test_image_checkpoint_preflight_fails_for_missing_required_checkpoint(tmp_path) -> None:
    config = {
        "display_name": "ResNet50",
        "backend": "torchvision",
        "architecture": "resnet50",
        "weights": None,
        "checkpoint_path": "checkpoints/resnet50/missing.pt",
    }

    with pytest.raises(FileNotFoundError, match="requires local checkpoint"):
        preflight_image_model_config(config, repo_root=tmp_path)


def test_image_checkpoint_preflight_allows_torchvision_hub_weights(tmp_path) -> None:
    report = preflight_image_model_config(
        {
            "display_name": "ResNet50",
            "backend": "torchvision",
            "architecture": "resnet50",
            "weights": "DEFAULT",
            "checkpoint_path": None,
        },
        repo_root=tmp_path,
    )

    assert report["status"] == "ok"
    assert report["checkpoint_required"] is False
    assert "no manual local checkpoint required" in report["notes"][0]


def test_image_checkpoint_preflight_rejects_non_image_only_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported image-only backend"):
        preflight_image_model_config(
            {
                "display_name": "MONET",
                "backend": "huggingface_clip",
                "checkpoint_path": None,
            },
            repo_root=tmp_path,
        )


def test_image_evaluation_uses_validation_selected_threshold_for_test_metrics() -> None:
    val_labels = [0, 0, 1, 1]
    val_probabilities = [0.05, 0.4, 0.45, 0.9]
    test_labels = [0, 1, 0, 1]
    test_probabilities = [0.44, 0.46, 0.2, 0.9]

    from isic2024_multimodal.evaluation.metrics import select_threshold_by_f1

    threshold = select_threshold_by_f1(val_labels, val_probabilities)
    test_metrics = evaluate_outputs(test_labels, test_probabilities, threshold=threshold)

    assert threshold == 0.45
    assert test_metrics["f1_score"] == 1.0
    assert test_metrics["false_positive_count"] == 0.0
    assert test_metrics["false_negative_count"] == 0.0


def test_image_trial_selection_score_never_falls_back_to_test_metrics() -> None:
    summary = {
        "best_validation_metrics": {
            "pauc_above_tpr80": float("nan"),
            "auc_roc": float("nan"),
            "average_precision": 0.25,
        },
        "test_metrics": {
            "average_precision": 0.99,
            "f1_score": 0.99,
        },
    }

    assert select_trial_score(summary) == 0.25


def test_image_suite_contains_requested_models_only() -> None:
    suite = json.loads(Path("experiments/configs/suites/image_baselines.json").read_text(encoding="utf-8"))

    assert suite["dataset_spec"] == "experiments/configs/dataset_specs/image_preprocessed_v1.json"
    assert suite["models"] == [
        "resnet50",
        "efficientnetv2_s",
        "convnextv2_tiny",
        "eva02_s",
        "vit_b",
        "edgenext_s",
    ]
    assert "efficientnet_b0" not in suite["models"]
    assert "monet" not in suite["models"]
    assert "biomedclip" not in suite["models"]


def test_selected_image_configs_are_pretrained_image_only_configs() -> None:
    suite = json.loads(Path("experiments/configs/suites/image_baselines.json").read_text(encoding="utf-8"))
    config_root = Path("experiments/configs/image_baselines")
    supported_backends = {"torchvision", "timm"}
    forbidden = {"iddx_full", "diagnosis", "diagnosis_text", "pathology_text"}

    for model_key in suite["models"]:
        config = json.loads((config_root / model_key / "config.json").read_text(encoding="utf-8"))
        model = config["model"]
        dataset = config["dataset"]
        serialized = json.dumps(config)
        assert model["backend"] in supported_backends
        assert model.get("checkpoint_path") is None
        assert model.get("weights") == "DEFAULT" or model.get("pretrained") is True
        assert dataset["sampler_strategy"] == "patient_balanced_weighted"
        assert "weighted_bce" in config["search_space"]["loss"]
        assert not any(column in serialized for column in forbidden)


def test_patient_balanced_sampler_weights_are_train_only_patient_normalized() -> None:
    samples = [
        types.SimpleNamespace(label=1, isic_id="A", group_id="P1", metadata={"patient_id": "P1"}),
        types.SimpleNamespace(label=1, isic_id="B", group_id="P1", metadata={"patient_id": "P1"}),
        types.SimpleNamespace(label=1, isic_id="C", group_id="P2", metadata={"patient_id": "P2"}),
        types.SimpleNamespace(label=0, isic_id="D", group_id="P3", metadata={"patient_id": "P3"}),
        types.SimpleNamespace(label=0, isic_id="E", group_id="P3", metadata={"patient_id": "P3"}),
        types.SimpleNamespace(label=0, isic_id="F", group_id="P4", metadata={"patient_id": "P4"}),
    ]

    weights = patient_balanced_sample_weights(samples)

    assert sum(weight for sample, weight in zip(samples, weights) if sample.label == 1) == pytest.approx(0.5)
    assert sum(weight for sample, weight in zip(samples, weights) if sample.label == 0) == pytest.approx(0.5)
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert weights[3] + weights[4] == pytest.approx(weights[5])


def test_image_binary_loss_uses_train_only_pos_weight() -> None:
    assert train_only_pos_weight([0, 0, 0, 1]) == 3.0

    class TinyDataset:
        samples = [
            types.SimpleNamespace(label=0),
            types.SimpleNamespace(label=0),
            types.SimpleNamespace(label=1),
        ]

    loader = types.SimpleNamespace(dataset=TinyDataset())
    loss_config = build_image_binary_loss_config(loader, hyperparameters={"loss": "weighted_bce"}, device="cpu")

    assert loss_config["loss_name"] == "weighted_bce"
    assert loss_config["train_positive_count"] == 1
    assert loss_config["train_negative_count"] == 2
    assert loss_config["pos_weight"] == 2.0


def test_image_one_logit_probabilities_use_sigmoid() -> None:
    import torch

    probabilities = positive_class_probabilities(torch.tensor([[-2.0], [0.0], [2.0]]))

    assert probabilities.tolist() == pytest.approx([0.1192029, 0.5, 0.8807970])


def test_run_all_image_models_cpu_policy_passes_cpu_device(tmp_path) -> None:
    config = tmp_path / "resnet50" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(
        dataset_root="data/raw/isic_2024_challenge",
        output_root="experiments/outputs/image_baselines",
        tracking_uri="file:experiments/logs/mlruns",
        experiment_name="ISIC2024-Image-Baselines",
        run_group_id="unit_group",
        dataset_id="image_preprocessed_v1",
        dataset_spec="experiments/configs/dataset_specs/image_preprocessed_v1.json",
        model_family="image_baselines",
        holdout_split_csv="data/splits/isic2024_train_validation_test_split_seed42.csv",
        cv_split_csv="data/splits/isic2024_train_validation_5fold_seed42.csv",
        cv_fold=0,
        seed=42,
        max_trials=None,
        epochs_override=None,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        batch_size_override=8,
        disable_pretrained=False,
        device_policy="cpu",
        devices=None,
        resolved_devices=[],
    )

    command = build_image_command(config, args, device=None)

    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--batch-size-override") + 1] == "8"


def test_cuda_cleanup_warning_does_not_raise(monkeypatch, capsys) -> None:
    import torch

    def fail_cleanup() -> None:
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", fail_cleanup)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None, raising=False)

    _cleanup_torch_state("cuda")

    assert "cuda cleanup warning: torch.cuda.empty_cache failed" in capsys.readouterr().out


def test_image_status_classifies_not_started_and_checkpoint_missing(tmp_path) -> None:
    hub_status = checkpoint_status_for_model(
        {
            "display_name": "ResNet-50",
            "backend": "torchvision",
            "architecture": "resnet50",
            "weights": "DEFAULT",
            "checkpoint_path": None,
        },
        repo_root=tmp_path,
    )
    missing_status = checkpoint_status_for_model(
        {
            "display_name": "ResNet50Custom",
            "backend": "torchvision",
            "architecture": "resnet50",
            "checkpoint_path": "checkpoints/resnet50/missing.pt",
        },
        repo_root=tmp_path,
    )

    assert classify_model_status(hub_status, {}) == "not_started"
    assert classify_model_status(missing_status, {}) == "checkpoint_missing"
    assert missing_status["status"] == "missing"


def test_image_status_classifies_preflight_failed(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "MONET" / "dummy.pt"
    checkpoint.parent.mkdir(parents=True)
    import torch

    torch.save({"state_dict": {"vision.weight": torch.ones(1)}}, checkpoint)
    status = checkpoint_status_for_model(
        {
            "display_name": "MONET",
            "backend": "huggingface_clip",
            "checkpoint_path": "checkpoints/MONET/dummy.pt",
        },
        repo_root=tmp_path,
    )

    assert status["status"] == "preflight_failed"
    assert classify_model_status(status, {}) == "preflight_failed"


def test_image_status_classifies_failed_and_smoke_passed(tmp_path) -> None:
    model_root = tmp_path / "ResNet-50"
    failed_trial = model_root / "ResNet-50_trial_001"
    failed_trial.mkdir(parents=True)
    (failed_trial / "error.txt").write_text("boom", encoding="utf-8")

    failed_status = artifact_status_for_model(model_root)
    assert failed_status["artifact_status"] == "failed"
    assert classify_model_status({"status": "hub/cache"}, failed_status) == "failed"

    passed_trial = model_root / "ResNet-50_trial_002"
    passed_trial.mkdir(parents=True)
    history = passed_trial / "history.csv"
    history.write_text("epoch,train_loss\n1,0.1\n", encoding="utf-8")
    write_json(
        passed_trial / "summary.json",
        {
            "history_path": str(history),
            "duration_seconds": 12.5,
            "best_validation_metric": 0.2,
        },
    )

    smoke_status = artifact_status_for_model(model_root)
    assert smoke_status["artifact_status"] == "smoke_passed"
    assert classify_model_status({"status": "hub/cache"}, smoke_status) == "smoke_passed"


def test_image_status_builds_records_from_suite(tmp_path) -> None:
    suite_path = tmp_path / "suite.json"
    config_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    write_json(
        suite_path,
        {
            "config_root": str(config_root),
            "models": ["resnet50", "resnet_missing"],
        },
    )
    write_json(
        config_root / "resnet50" / "config.json",
        {
            "model": {
                "display_name": "ResNet-50",
                "backend": "torchvision",
                "architecture": "resnet50",
                "weights": "DEFAULT",
                "checkpoint_path": None,
            }
        },
    )
    write_json(
        config_root / "resnet_missing" / "config.json",
        {
            "model": {
                "display_name": "ResNet50Custom",
                "backend": "torchvision",
                "architecture": "resnet50",
                "checkpoint_path": "checkpoints/resnet50/missing.pt",
            }
        },
    )

    records = build_status_records(suite_path=suite_path, output_root=output_root, repo_root=tmp_path)
    by_model = {record["model_key"]: record for record in records}

    assert by_model["resnet50"]["status"] == "not_started"
    assert by_model["resnet50"]["checkpoint_status"] == "hub/cache"
    assert by_model["resnet_missing"]["status"] == "checkpoint_missing"
    assert by_model["resnet_missing"]["checkpoint_status"] == "missing"


def test_eta_duration_formatting() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(65) == "1m 5s"
    assert format_duration(3661) == "1h 1m 1s"


def test_family_runner_builds_tabular_family_command(tmp_path) -> None:
    feature_set_json = tmp_path / "features.json"
    write_json(
        feature_set_json,
        {
            "target_column": "target",
            "feature_sets": {"strict_main_input": ["age_approx"]},
        },
    )
    spec_path = tmp_path / "dataset_spec.json"
    write_json(
        spec_path,
        {
            "dataset_id": "strict_main_input_v1",
            "dataset_root": "data/raw",
            "feature_set_json": str(feature_set_json),
            "feature_sets": ["strict_main_input"],
        },
    )
    dataset_spec = load_dataset_spec(spec_path, repo_root=tmp_path)
    paths = resolve_family_paths(
        family="tabular_baselines",
        run_group_id="run_a",
        repo_root=tmp_path,
    )
    args = types.SimpleNamespace(
        run_group_id="run_a",
        tracking_uri="file:experiments/logs/mlruns",
        devices=[0],
        device_policy="auto",
        smoke=True,
        skip_reports=True,
    )

    command = build_tabular_command(
        suite={"models": ["xgboost"], "feature_sets": ["strict_main_input"], "smoke": {"max_train": 10}},
        dataset_spec=dataset_spec,
        paths=paths,
        args=args,
    )

    assert "isic2024_multimodal.cli.run_all_tabular_models" in command
    assert command[command.index("--dataset-id") + 1] == "strict_main_input_v1"
    assert command[command.index("--model-family") + 1] == "tabular_baselines"
    assert command[command.index("--output-root") + 1] == str(paths.output_root)
    assert command[command.index("--max-train-rows") + 1] == "10"
    assert command[command.index("--device-policy") + 1] == "auto"


def test_baseline_suite_builds_tabular_and_image_family_commands() -> None:
    args = types.SimpleNamespace(
        families=["tabular_baselines", "image_baselines"],
        run_group_id="baseline_suite_unit",
        devices=[0, 1],
        device_policy="cpu",
        smoke=True,
        preflight_only=True,
        resume=True,
        reset_family_output=False,
        skip_reports=True,
    )

    commands = build_suite_commands(args)

    assert [entry.family for entry in commands] == ["tabular_baselines", "image_baselines"]
    for entry in commands:
        command = entry.command
        assert "isic2024_multimodal.cli.run_experiment_family" in command
        assert command[command.index("--family") + 1] == entry.family
        assert command[command.index("--run-group-id") + 1] == "baseline_suite_unit"
        assert command[command.index("--device-policy") + 1] == "cpu"
        assert command[command.index("--devices") + 1 : command.index("--devices") + 3] == ["0", "1"]
        assert "--smoke" in command
        assert "--preflight-only" in command
        assert "--resume" in command
        assert "--skip-reports" in command
