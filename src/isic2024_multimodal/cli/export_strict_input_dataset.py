from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isic2024_multimodal.data.tabular_dataset import DEFAULT_DATASET_ROOT, DEFAULT_TARGET_COLUMN, load_tabular_dataframe
from isic2024_multimodal.data.triple_stratified_split import build_nested_cv_assignments


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
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--sample-count-bins", type=int, default=5)
    parser.add_argument("--strict-output", default="data/processed/isic2024_strict_model_input.csv")
    parser.add_argument("--iddx-sidecar-output", default="data/processed/isic2024_iddx_full_train_only_sidecar.csv")
    parser.add_argument(
        "--nested-split-output",
        default=None,
        help="Defaults to data/splits/isic2024_official_train_nested_{outer}x{inner}_seed{seed}.csv",
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

    split_results = build_nested_cv_assignments(
        frame,
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        sample_count_bins=args.sample_count_bins,
    )

    strict_frame = build_strict_model_input(frame)
    iddx_sidecar_frame = build_iddx_sidecar(frame)
    nested_split_frame = build_nested_split_frame(frame, split_results)

    summary = build_export_summary(
        frame=frame,
        strict_frame=strict_frame,
        iddx_sidecar_frame=iddx_sidecar_frame,
        nested_split_frame=nested_split_frame,
        split_results=split_results,
        args=args,
        output_paths=output_paths,
    )
    validate_export_contract(summary)

    write_csv(strict_frame, output_paths["strict_output"])
    write_csv(iddx_sidecar_frame, output_paths["iddx_sidecar_output"])
    write_csv(nested_split_frame, output_paths["nested_split_output"])
    write_json(summary, output_paths["summary_output"])

    print(f"Saved strict input dataset to {output_paths['strict_output']}")
    print(f"Saved train-only iddx sidecar to {output_paths['iddx_sidecar_output']}")
    print(f"Saved nested CV split to {output_paths['nested_split_output']}")
    print(f"Saved validation protocol summary to {output_paths['summary_output']}")


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    seed = args.seed
    return {
        "strict_output": Path(args.strict_output),
        "iddx_sidecar_output": Path(args.iddx_sidecar_output),
        "nested_split_output": Path(
            args.nested_split_output
            or f"data/splits/isic2024_official_train_nested_{args.outer_folds}x{args.inner_folds}_seed{seed}.csv"
        ),
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


def build_nested_split_frame(frame, split_results):
    rows = []
    base_frame = frame[IDENTIFIER_COLUMNS].copy()
    patient_ids = frame["patient_id"].astype(str)
    outer_assignment = split_results.outer.patient_assignment
    for outer_fold in range(split_results.outer_folds):
        inner_assignment = split_results.inner_by_outer_fold[outer_fold].patient_assignment
        for inner_fold in range(split_results.inner_folds):
            role = patient_ids.map(
                lambda patient_id: nested_split_role(
                    patient_id,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    outer_assignment=outer_assignment,
                    inner_assignment=inner_assignment,
                )
            )
            role_frame = base_frame.copy()
            role_frame["outer_fold"] = outer_fold
            role_frame["cv_test_fold"] = outer_fold
            role_frame["inner_fold"] = inner_fold
            role_frame["split_role"] = role
            rows.append(role_frame)
    import pandas as pd

    return pd.concat(rows, ignore_index=True)


def nested_split_role(
    patient_id: str,
    *,
    outer_fold: int,
    inner_fold: int,
    outer_assignment: dict[str, int],
    inner_assignment: dict[str, int],
) -> str:
    if int(outer_assignment[patient_id]) == int(outer_fold):
        return "outer_test"
    if int(inner_assignment[patient_id]) == int(inner_fold):
        return "inner_validation"
    return "inner_train"


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
    nested_split_frame,
    split_results,
    args: argparse.Namespace,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    nested_summary = summarize_nested_split(frame, nested_split_frame)
    overlap_summary = summarize_nested_patient_overlap(nested_split_frame)
    disallowed_present = sorted(set(strict_frame.columns) & DISALLOWED_MAIN_COLUMNS)
    missingness_summary = summarize_missingness(frame)

    return {
        "dataset_name": "isic2024_strict_input_iddx_full_contract",
        "protocol_version": "patient_level_triple_stratified_nested_cv_v1",
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
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
            "outer_balance_score": split_results.outer.balance_score,
            "inner_balance_scores": {
                str(outer_fold): inner_result.balance_score
                for outer_fold, inner_result in split_results.inner_by_outer_fold.items()
            },
        },
        "nested_cv_summary": nested_summary,
        "missingness_summary": missingness_summary,
        "overlap_summary": overlap_summary,
        "leakage_controls": {
            "patient_disjoint_outer_cv": all(
                item["cv_train_outer_test_patient_overlap"] == 0 for item in overlap_summary["outer"]
            ),
            "patient_disjoint_inner_cv": all(
                item["inner_train_inner_validation_patient_overlap"] == 0
                and item["inner_train_outer_test_patient_overlap"] == 0
                and item["inner_validation_outer_test_patient_overlap"] == 0
                for item in overlap_summary["inner"]
            ),
            "triple_stratified_outer_folds": True,
            "triple_stratified_inner_folds": True,
            "iddx_full_excluded_from_strict_input": "iddx_full" not in strict_frame.columns,
            "diagnosis_reference_columns_excluded_from_strict_input": len(disallowed_present) == 0,
            "train_only_preprocessing_performed": False,
        },
    }


def summarize_missingness(frame) -> list[dict[str, Any]]:
    row_count = max(int(len(frame)), 1)
    rows = []
    column_roles = {
        **{column: "identifier" for column in IDENTIFIER_COLUMNS},
        **{column: "feature" for column in STRICT_INPUT_COLUMNS},
        **{column: "privileged_excluded" for column in DISALLOWED_MAIN_COLUMNS if column in frame.columns},
    }
    for column, role in column_roles.items():
        if column not in frame.columns:
            continue
        missing_count = int(frame[column].isna().sum())
        if missing_count == 0:
            continue
        rows.append(
            {
                "column": column,
                "role": role,
                "missing_count": missing_count,
                "missing_rate": missing_count / row_count,
            }
        )
    return sorted(rows, key=lambda item: (-item["missing_count"], item["column"]))


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


def summarize_nested_split(frame, nested_split_frame) -> dict[str, list[dict[str, Any]]]:
    merged = nested_split_frame.merge(
        frame[IDENTIFIER_COLUMNS + [DEFAULT_TARGET_COLUMN]],
        on=IDENTIFIER_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    outer_source = merged.assign(
        outer_role=merged["split_role"].where(
            merged["split_role"].eq("outer_test"),
            "cv_train",
        )
    )
    outer_source = outer_source.drop_duplicates(["isic_id", "outer_fold", "outer_role"])
    outer_summary = (
        outer_source.groupby(["outer_fold", "outer_role"], dropna=False)
        .agg(
            rows=("isic_id", "nunique"),
            patients=("patient_id", "nunique"),
            positive_rows=(DEFAULT_TARGET_COLUMN, "sum"),
        )
        .reset_index()
        .sort_values(["outer_fold", "outer_role"])
    )
    outer_summary["positive_rate_pct"] = (outer_summary["positive_rows"] / outer_summary["rows"] * 100).round(5)

    inner_source = merged.loc[~merged["split_role"].eq("outer_test")].copy()
    inner_summary = (
        inner_source.groupby(["outer_fold", "inner_fold", "split_role"], dropna=False)
        .agg(
            rows=("isic_id", "nunique"),
            patients=("patient_id", "nunique"),
            positive_rows=(DEFAULT_TARGET_COLUMN, "sum"),
        )
        .reset_index()
        .sort_values(["outer_fold", "inner_fold", "split_role"])
    )
    inner_summary["positive_rate_pct"] = (inner_summary["positive_rows"] / inner_summary["rows"] * 100).round(5)
    return {
        "outer": outer_summary.to_dict(orient="records"),
        "inner": inner_summary.to_dict(orient="records"),
    }


def summarize_nested_patient_overlap(nested_split_frame) -> dict[str, Any]:
    outer_rows = []
    inner_rows = []
    for outer_fold, outer_frame in nested_split_frame.groupby("outer_fold", dropna=False):
        outer_test_patients = set(
            outer_frame.loc[outer_frame["split_role"].eq("outer_test"), "patient_id"].astype(str)
        )
        cv_train_patients = set(
            outer_frame.loc[~outer_frame["split_role"].eq("outer_test"), "patient_id"].astype(str)
        )
        outer_rows.append(
            {
                "outer_fold": int(outer_fold),
                "cv_train_outer_test_patient_overlap": len(cv_train_patients & outer_test_patients),
            }
        )
        for inner_fold, inner_frame in outer_frame.groupby("inner_fold", dropna=False):
            inner_train_patients = set(
                inner_frame.loc[inner_frame["split_role"].eq("inner_train"), "patient_id"].astype(str)
            )
            inner_validation_patients = set(
                inner_frame.loc[inner_frame["split_role"].eq("inner_validation"), "patient_id"].astype(str)
            )
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "inner_train_inner_validation_patient_overlap": len(inner_train_patients & inner_validation_patients),
                    "inner_train_outer_test_patient_overlap": len(inner_train_patients & outer_test_patients),
                    "inner_validation_outer_test_patient_overlap": len(inner_validation_patients & outer_test_patients),
                }
            )
    return {"outer": outer_rows, "inner": inner_rows}


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
