from __future__ import annotations

import pandas as pd

from isic2024_multimodal.cli.export_strict_input_dataset import (
    DISALLOWED_MAIN_COLUMNS,
    MAIN_OUTPUT_COLUMNS,
    STRICT_INPUT_COLUMNS,
    build_cv_split_frame,
    build_holdout_split_frame,
    build_iddx_sidecar,
    build_strict_model_input,
    summarize_patient_overlap,
    validate_source_frame,
)
from isic2024_multimodal.data.triple_stratified_split import build_holdout_and_cv_assignments


def make_synthetic_frame() -> pd.DataFrame:
    rows = []
    for patient_index in range(20):
        patient_id = f"P{patient_index:03d}"
        for sample_index in range(3 + patient_index % 4):
            row = {
                "isic_id": f"ISIC_{patient_index:03d}_{sample_index:02d}",
                "patient_id": patient_id,
                "lesion_id": f"L{patient_index:03d}_{sample_index % 2}",
                "target": int(patient_index % 5 == 0 and sample_index == 0),
                "iddx_full": "Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma"
                if patient_index % 5 == 0 and sample_index == 0
                else "Benign",
            }
            for column in STRICT_INPUT_COLUMNS:
                row[column] = f"{column}_category" if column in {"sex", "anatom_site_general", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"} else float(patient_index + sample_index)
            for column in DISALLOWED_MAIN_COLUMNS - {"iddx_full"}:
                row[column] = "reference"
            rows.append(row)
    return pd.DataFrame(rows)


def test_strict_input_columns_exist_and_exclude_iddx_reference_columns() -> None:
    frame = make_synthetic_frame()
    validate_source_frame(frame)

    strict_frame = build_strict_model_input(frame)

    assert list(strict_frame.columns) == MAIN_OUTPUT_COLUMNS
    assert set(STRICT_INPUT_COLUMNS).issubset(strict_frame.columns)
    assert not (set(strict_frame.columns) & DISALLOWED_MAIN_COLUMNS)


def test_iddx_sidecar_is_one_to_one_aligned_with_main_table() -> None:
    frame = make_synthetic_frame()

    strict_frame = build_strict_model_input(frame)
    sidecar_frame = build_iddx_sidecar(frame)

    assert "iddx_full_train_only" in sidecar_frame.columns
    assert "iddx_full" not in sidecar_frame.columns
    assert strict_frame["isic_id"].equals(sidecar_frame["isic_id"])
    assert sidecar_frame["isic_id"].is_unique


def test_patient_disjoint_holdout_and_cv_splits() -> None:
    frame = make_synthetic_frame()
    split_results = build_holdout_and_cv_assignments(frame, seed=42, test_size=0.2, cv_folds=5, sample_count_bins=5)

    holdout_split_frame = build_holdout_split_frame(frame, split_results["holdout"].patient_assignment)
    cv_split_frame = build_cv_split_frame(frame, holdout_split_frame, split_results["cv"].patient_assignment)
    overlap_summary = summarize_patient_overlap(holdout_split_frame, cv_split_frame)

    assert overlap_summary["train_validation_test_patient_overlap"] == 0
    assert len(cv_split_frame["cv_validation_fold"].unique()) == 5
    for fold_summary in overlap_summary["cv"]:
        assert fold_summary["cv_train_cv_validation_patient_overlap"] == 0
        assert fold_summary["cv_validation_test_data_patient_overlap"] == 0
        assert fold_summary["cv_train_test_data_patient_overlap"] == 0


def test_triple_stratified_split_is_deterministic_for_same_seed() -> None:
    frame = make_synthetic_frame()

    first = build_holdout_and_cv_assignments(frame, seed=42, test_size=0.2, cv_folds=5, sample_count_bins=5)
    second = build_holdout_and_cv_assignments(frame, seed=42, test_size=0.2, cv_folds=5, sample_count_bins=5)

    assert first["holdout"].patient_assignment == second["holdout"].patient_assignment
    assert first["cv"].patient_assignment == second["cv"].patient_assignment
