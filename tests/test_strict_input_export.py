from __future__ import annotations

import pandas as pd

from isic2024_multimodal.cli.export_strict_input_dataset import (
    DISALLOWED_MAIN_COLUMNS,
    MAIN_OUTPUT_COLUMNS,
    STRICT_INPUT_COLUMNS,
    build_iddx_sidecar,
    build_nested_split_frame,
    build_strict_model_input,
    summarize_missingness,
    summarize_nested_patient_overlap,
    validate_source_frame,
)
from isic2024_multimodal.data.triple_stratified_split import build_nested_cv_assignments


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


def test_missingness_summary_records_roles_without_imputation() -> None:
    frame = make_synthetic_frame()
    frame.loc[0, "age_approx"] = None
    frame.loc[1, "sex"] = None
    frame.loc[2, "lesion_id"] = None
    frame.loc[3, "iddx_full"] = None

    summary = summarize_missingness(frame)
    by_column = {item["column"]: item for item in summary}

    assert by_column["age_approx"]["role"] == "feature"
    assert by_column["sex"]["role"] == "feature"
    assert by_column["lesion_id"]["role"] == "identifier"
    assert by_column["iddx_full"]["role"] == "privileged_excluded"
    assert pd.isna(frame.loc[0, "age_approx"])


def test_patient_disjoint_nested_cv_splits() -> None:
    frame = make_synthetic_frame()
    split_results = build_nested_cv_assignments(frame, seed=42, outer_folds=5, inner_folds=4, sample_count_bins=5)

    nested_split_frame = build_nested_split_frame(frame, split_results)
    overlap_summary = summarize_nested_patient_overlap(nested_split_frame)

    assert set(nested_split_frame["split_role"].unique()) == {"inner_train", "inner_validation", "outer_test"}
    assert nested_split_frame["outer_fold"].nunique() == 5
    assert nested_split_frame["inner_fold"].nunique() == 4
    for fold_summary in overlap_summary["outer"]:
        assert fold_summary["cv_train_outer_test_patient_overlap"] == 0
    for fold_summary in overlap_summary["inner"]:
        assert fold_summary["inner_train_inner_validation_patient_overlap"] == 0
        assert fold_summary["inner_train_outer_test_patient_overlap"] == 0
        assert fold_summary["inner_validation_outer_test_patient_overlap"] == 0


def test_triple_stratified_split_is_deterministic_for_same_seed() -> None:
    frame = make_synthetic_frame()

    first = build_nested_cv_assignments(frame, seed=42, outer_folds=5, inner_folds=4, sample_count_bins=5)
    second = build_nested_cv_assignments(frame, seed=42, outer_folds=5, inner_folds=4, sample_count_bins=5)

    assert first.outer.patient_assignment == second.outer.patient_assignment
    for outer_fold in range(first.outer_folds):
        assert first.inner_by_outer_fold[outer_fold].patient_assignment == second.inner_by_outer_fold[outer_fold].patient_assignment
