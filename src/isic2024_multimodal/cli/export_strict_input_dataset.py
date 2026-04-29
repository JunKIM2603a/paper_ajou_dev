from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isic2024_multimodal.data.tabular_dataset import DEFAULT_DATASET_ROOT, DEFAULT_TARGET_COLUMN, load_tabular_dataframe
from isic2024_multimodal.data.triple_stratified_split import build_holdout_and_cv_assignments


STRICT_INPUT_COLUMNS = [
    "age_approx",
    "sex",
    "anatom_site_general",
    "clin_size_long_diam_mm",
    "tbp_tile_type",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
]

IDENTIFIER_COLUMNS = ["isic_id", "patient_id", "lesion_id"]
MAIN_OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN] + STRICT_INPUT_COLUMNS
SIDECAR_OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN, "iddx_full_train_only"]
DISALLOWED_MAIN_COLUMNS = {
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "mel_mitotic_index",
    "mel_thick_mm",
    "tbp_lv_dnn_lesion_confidence",
    "attribution",
    "copyright_license",
    "image_type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the ISIC2024 strict_input dataset, train-only iddx sidecar, and patient-level splits."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--sample-count-bins", type=int, default=5)
    parser.add_argument("--strict-output", default="data/processed/isic2024_strict_model_input.csv")
    parser.add_argument("--iddx-sidecar-output", default="data/processed/isic2024_iddx_full_train_only_sidecar.csv")
    parser.add_argument(
        "--holdout-split-output",
        default=None,
        help="Defaults to data/splits/isic2024_train_validation_test_split_seed{seed}.csv",
    )
    parser.add_argument(
        "--cv-split-output",
        default=None,
        help="Defaults to data/splits/isic2024_train_validation_5fold_seed{seed}.csv",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Defaults to experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed{seed}.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = resolve_output_paths(args)
    frame = load_tabular_dataframe(args.dataset_root, include_image_columns=False)
    validate_source_frame(frame)

    split_results = build_holdout_and_cv_assignments(
        frame,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        sample_count_bins=args.sample_count_bins,
    )

    strict_frame = build_strict_model_input(frame)
    iddx_sidecar_frame = build_iddx_sidecar(frame)
    holdout_split_frame = build_holdout_split_frame(frame, split_results["holdout"].patient_assignment)
    cv_split_frame = build_cv_split_frame(frame, holdout_split_frame, split_results["cv"].patient_assignment)

    summary = build_export_summary(
        frame=frame,
        strict_frame=strict_frame,
        iddx_sidecar_frame=iddx_sidecar_frame,
        holdout_split_frame=holdout_split_frame,
        cv_split_frame=cv_split_frame,
        split_results=split_results,
        args=args,
        output_paths=output_paths,
    )
    validate_export_contract(summary)

    write_csv(strict_frame, output_paths["strict_output"])
    write_csv(iddx_sidecar_frame, output_paths["iddx_sidecar_output"])
    write_csv(holdout_split_frame, output_paths["holdout_split_output"])
    write_csv(cv_split_frame, output_paths["cv_split_output"])
    write_json(summary, output_paths["summary_output"])

    print(f"Saved strict input dataset to {output_paths['strict_output']}")
    print(f"Saved train-only iddx sidecar to {output_paths['iddx_sidecar_output']}")
    print(f"Saved holdout split to {output_paths['holdout_split_output']}")
    print(f"Saved CV split to {output_paths['cv_split_output']}")
    print(f"Saved validation protocol summary to {output_paths['summary_output']}")


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    seed = args.seed
    return {
        "strict_output": Path(args.strict_output),
        "iddx_sidecar_output": Path(args.iddx_sidecar_output),
        "holdout_split_output": Path(
            args.holdout_split_output or f"data/splits/isic2024_train_validation_test_split_seed{seed}.csv"
        ),
        "cv_split_output": Path(args.cv_split_output or f"data/splits/isic2024_train_validation_5fold_seed{seed}.csv"),
        "summary_output": Path(
            args.summary_output
            or f"experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed{seed}.json"
        ),
    }


def validate_source_frame(frame) -> None:
    required_columns = MAIN_OUTPUT_COLUMNS + ["iddx_full"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required strict_input export columns are missing: {missing_columns}")
    if not frame["isic_id"].is_unique:
        raise ValueError("isic_id must be unique before export.")
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id must be present for patient-level split.")
    observed_targets = set(frame[DEFAULT_TARGET_COLUMN].dropna().astype(int).unique().tolist())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(f"target must be binary, observed {sorted(observed_targets)}")


def build_strict_model_input(frame):
    strict_frame = frame[MAIN_OUTPUT_COLUMNS].copy()
    leaked_columns = sorted(set(strict_frame.columns) & DISALLOWED_MAIN_COLUMNS)
    if leaked_columns:
        raise RuntimeError(f"Disallowed columns leaked into strict input export: {leaked_columns}")
    return strict_frame


def build_iddx_sidecar(frame):
    sidecar_frame = frame[IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN, "iddx_full"]].copy()
    return sidecar_frame.rename(columns={"iddx_full": "iddx_full_train_only"})[SIDECAR_OUTPUT_COLUMNS]


def build_holdout_split_frame(frame, assignment: dict[str, int]):
    split_frame = frame[IDENTIFIER_COLUMNS].copy()
    split_frame["split"] = frame["patient_id"].astype(str).map(
        lambda patient_id: "train_validation_data" if assignment[patient_id] == 0 else "test_data"
    )
    return split_frame


def build_cv_split_frame(frame, holdout_split_frame, assignment: dict[str, int]):
    train_validation_mask = holdout_split_frame["split"].eq("train_validation_data")
    cv_frame = frame.loc[train_validation_mask, IDENTIFIER_COLUMNS].copy()
    cv_frame["cv_validation_fold"] = frame.loc[train_validation_mask, "patient_id"].astype(str).map(assignment).astype(int)
    return cv_frame


def build_export_summary(
    *,
    frame,
    strict_frame,
    iddx_sidecar_frame,
    holdout_split_frame,
    cv_split_frame,
    split_results,
    args: argparse.Namespace,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    holdout_counts = summarize_holdout_split(frame, holdout_split_frame)
    cv_counts = summarize_cv_split(frame, cv_split_frame)
    overlap_summary = summarize_patient_overlap(holdout_split_frame, cv_split_frame)
    disallowed_present = sorted(set(strict_frame.columns) & DISALLOWED_MAIN_COLUMNS)

    return {
        "dataset_name": "isic2024_strict_input_iddx_full_contract",
        "seed": args.seed,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "sample_count_bins": args.sample_count_bins,
        "source": {
            "dataset_root": str(Path(args.dataset_root)),
            "row_count": int(len(frame)),
            "patient_count": int(frame["patient_id"].nunique()),
            "isic_id_unique": bool(frame["isic_id"].is_unique),
            "positive_rows": int(frame[DEFAULT_TARGET_COLUMN].sum()),
        },
        "outputs": {key: str(path) for key, path in output_paths.items()},
        "strict_input_contract": {
            "identifier_columns": IDENTIFIER_COLUMNS,
            "target_column": DEFAULT_TARGET_COLUMN,
            "strict_input_columns": STRICT_INPUT_COLUMNS,
            "strict_input_column_count": len(STRICT_INPUT_COLUMNS),
            "disallowed_main_columns_present": disallowed_present,
            "inference_requires_iddx_full": False,
        },
        "sidecar_contract": {
            "columns": SIDECAR_OUTPUT_COLUMNS,
            "role": "train-only privileged supervision candidate sidecar",
            "ordinary_inference_input": False,
            "row_aligned_with_strict_input": bool(strict_frame["isic_id"].equals(iddx_sidecar_frame["isic_id"])),
        },
        "split_scores": {
            "holdout_balance_score": split_results["holdout"].balance_score,
            "cv_balance_score": split_results["cv"].balance_score,
        },
        "holdout_summary": holdout_counts,
        "cv_summary": cv_counts,
        "overlap_summary": overlap_summary,
        "leakage_controls": {
            "patient_disjoint_holdout": overlap_summary["train_validation_test_patient_overlap"] == 0,
            "patient_disjoint_cv": all(item["cv_train_cv_validation_patient_overlap"] == 0 for item in overlap_summary["cv"]),
            "iddx_full_excluded_from_strict_input": "iddx_full" not in strict_frame.columns,
            "diagnosis_reference_columns_excluded_from_strict_input": len(disallowed_present) == 0,
            "train_only_preprocessing_performed": False,
        },
    }


def summarize_holdout_split(frame, holdout_split_frame) -> list[dict[str, Any]]:
    merged = frame[IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN]].merge(
        holdout_split_frame[["isic_id", "split"]],
        on="isic_id",
        how="inner",
        validate="1:1",
    )
    summary = (
        merged.groupby("split", dropna=False)
        .agg(
            rows=("isic_id", "size"),
            patients=("patient_id", "nunique"),
            positive_rows=(DEFAULT_TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )
    summary["positive_rate_pct"] = (summary["positive_rows"] / summary["rows"] * 100).round(5)
    return summary.to_dict(orient="records")


def summarize_cv_split(frame, cv_split_frame) -> list[dict[str, Any]]:
    merged = frame[IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN]].merge(
        cv_split_frame[["isic_id", "cv_validation_fold"]],
        on="isic_id",
        how="inner",
        validate="1:1",
    )
    summary = (
        merged.groupby("cv_validation_fold", dropna=False)
        .agg(
            rows=("isic_id", "size"),
            patients=("patient_id", "nunique"),
            positive_rows=(DEFAULT_TARGET_COLUMN, "sum"),
        )
        .reset_index()
        .sort_values("cv_validation_fold")
    )
    summary["positive_rate_pct"] = (summary["positive_rows"] / summary["rows"] * 100).round(5)
    return summary.to_dict(orient="records")


def summarize_patient_overlap(holdout_split_frame, cv_split_frame) -> dict[str, Any]:
    train_validation_patients = set(
        holdout_split_frame.loc[holdout_split_frame["split"].eq("train_validation_data"), "patient_id"].astype(str)
    )
    test_patients = set(holdout_split_frame.loc[holdout_split_frame["split"].eq("test_data"), "patient_id"].astype(str))
    cv_overlap = []
    all_cv_patients = set(cv_split_frame["patient_id"].astype(str))
    for fold in sorted(cv_split_frame["cv_validation_fold"].unique().tolist()):
        validation_patients = set(
            cv_split_frame.loc[cv_split_frame["cv_validation_fold"].eq(fold), "patient_id"].astype(str)
        )
        train_patients = all_cv_patients - validation_patients
        cv_overlap.append(
            {
                "cv_fold": int(fold),
                "cv_train_cv_validation_patient_overlap": len(train_patients & validation_patients),
                "cv_validation_test_data_patient_overlap": len(validation_patients & test_patients),
                "cv_train_test_data_patient_overlap": len(train_patients & test_patients),
            }
        )
    return {
        "train_validation_test_patient_overlap": len(train_validation_patients & test_patients),
        "cv": cv_overlap,
    }


def validate_export_contract(summary: dict[str, Any]) -> None:
    leakage_controls = summary["leakage_controls"]
    failed_controls = [name for name, passed in leakage_controls.items() if passed is not True and name != "train_only_preprocessing_performed"]
    if failed_controls:
        raise RuntimeError(f"Strict input export failed leakage controls: {failed_controls}")


def write_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
