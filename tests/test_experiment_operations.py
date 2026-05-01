from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from isic2024_multimodal.cli.run_experiment_family import build_tabular_command
from isic2024_multimodal.data.image_dataset import create_splits_from_locked_csvs
from isic2024_multimodal.experiments.dataset_specs import load_dataset_spec
from isic2024_multimodal.experiments.families import reset_family_outputs, resolve_family_paths
from isic2024_multimodal.experiments.registry import read_selection_registry, write_family_selection
from isic2024_multimodal.reporting.mlflow_report import build_filter_string


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

