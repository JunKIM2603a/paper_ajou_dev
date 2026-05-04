from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from isic2024_multimodal.cli.run_image_baseline import select_trial_score
from isic2024_multimodal.cli.run_all_image_models import build_command as build_image_command
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
from isic2024_multimodal.experiments.registry import read_selection_registry, write_family_selection
from isic2024_multimodal.models.image.checkpoint_preflight import preflight_image_model_config
from isic2024_multimodal.models.image.checkpoint_downloads import download_checkpoint, infer_model_key
from isic2024_multimodal.reporting.mlflow_report import build_filter_string
from isic2024_multimodal.training.trainer import format_duration
from isic2024_multimodal.training.trainer import evaluate_outputs


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
        "display_name": "CheXzero",
        "backend": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": None,
        "checkpoint_path": "checkpoints/CheXzero/missing.pt",
    }

    with pytest.raises(FileNotFoundError, match="requires local checkpoint"):
        preflight_image_model_config(config, repo_root=tmp_path)


def test_image_checkpoint_preflight_allows_torchvision_hub_weights(tmp_path) -> None:
    report = preflight_image_model_config(
        {
            "display_name": "ResNet-50",
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


def test_image_checkpoint_preflight_rejects_medclip_sam_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "MedCLIP" / "sam_vit_b_01ec64.pth"
    checkpoint.parent.mkdir(parents=True)
    import torch

    torch.save({"state_dict": {"vision.weight": torch.ones(1)}}, checkpoint)

    with pytest.raises(ValueError, match="SAM checkpoint"):
        preflight_image_model_config(
            {
                "display_name": "MedCLIP",
                "backend": "medclip",
                "architecture": "vit",
                "pretrained": False,
                "checkpoint_path": "checkpoints/MedCLIP/sam_vit_b_01ec64.pth",
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

    assert suite["models"] == [
        "biomedclip",
        "chexzero",
        "deit_s",
        "densenet121",
        "dinov2_vit_s",
        "efficientnet_b0",
        "eyepacs",
        "medclip",
        "resnet50",
        "retfound",
        "torchxrayvision",
        "vit_b_16",
    ]
    assert "ham10000" not in suite["models"]
    assert "monet" not in suite["models"]


def test_image_download_registry_skips_models_without_manual_checkpoint(tmp_path) -> None:
    report = download_checkpoint("resnet50", repo_root=tmp_path)

    assert report["status"] == "skipped"


def test_image_download_model_key_inference_for_required_checkpoints() -> None:
    assert infer_model_key({"display_name": "CheXzero"}) == "chexzero"
    assert infer_model_key({"display_name": "EyePACS"}) == "eyepacs"
    assert infer_model_key({"display_name": "MedCLIP", "backend": "medclip"}) == "medclip"
    assert infer_model_key({"display_name": "RETFound"}) == "retfound"


def test_run_all_image_models_passes_auto_download_flag(tmp_path) -> None:
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
        max_trials=1,
        epochs_override=1,
        max_train_samples=10,
        max_val_samples=10,
        max_test_samples=10,
        disable_pretrained=False,
        auto_download_checkpoints=True,
    )

    command = build_image_command(config, args, device=None)

    assert "--auto-download-checkpoints" in command


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
            "display_name": "CheXzero",
            "backend": "open_clip",
            "checkpoint_path": "checkpoints/CheXzero/missing.pt",
        },
        repo_root=tmp_path,
    )

    assert classify_model_status(hub_status, {}) == "not_started"
    assert classify_model_status(missing_status, {}) == "checkpoint_missing"
    assert missing_status["status"] == "downloadable"


def test_image_status_classifies_preflight_failed(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "MedCLIP" / "sam_vit_b_01ec64.pth"
    checkpoint.parent.mkdir(parents=True)
    import torch

    torch.save({"state_dict": {"vision.weight": torch.ones(1)}}, checkpoint)
    status = checkpoint_status_for_model(
        {
            "display_name": "MedCLIP",
            "backend": "medclip",
            "architecture": "vit",
            "pretrained": False,
            "checkpoint_path": "checkpoints/MedCLIP/sam_vit_b_01ec64.pth",
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
            "models": ["resnet50", "chexzero"],
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
        config_root / "chexzero" / "config.json",
        {
            "model": {
                "display_name": "CheXzero",
                "backend": "open_clip",
                "checkpoint_path": "checkpoints/CheXzero/missing.pt",
            }
        },
    )

    records = build_status_records(suite_path=suite_path, output_root=output_root, repo_root=tmp_path)
    by_model = {record["model_key"]: record for record in records}

    assert by_model["resnet50"]["status"] == "not_started"
    assert by_model["resnet50"]["checkpoint_status"] == "hub/cache"
    assert by_model["chexzero"]["status"] == "checkpoint_missing"
    assert by_model["chexzero"]["checkpoint_status"] == "downloadable"


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
