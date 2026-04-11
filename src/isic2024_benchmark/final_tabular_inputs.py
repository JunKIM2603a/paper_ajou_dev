from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isic2024_benchmark.tabular_terms import (
    LEGACY_STRICT_FULL,
    RELAXED,
    STRICT_BASE,
    STRICT_FE,
    STRICT_MAIN_INPUT,
    normalize_feature_set_name,
)

FINAL_INPUTS_FEATURE_SET_SOURCE = "final_inputs/final_feature_sets_v3.json"


def load_strict_preprocessing_spec(eda_dir: str | Path) -> dict:
    path = Path(eda_dir) / "preprocessing" / "strict_preprocessing_spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_feature_sets_v3(eda_dir: str | Path) -> dict:
    path = Path(eda_dir) / "final_inputs" / "final_feature_sets_v3.json"
    return json.loads(path.read_text(encoding="utf-8"))


def is_final_inputs_feature_payload(feature_payload: dict) -> bool:
    evidence = feature_payload.get("evidence", {})
    return evidence.get("feature_sets_source") == FINAL_INPUTS_FEATURE_SET_SOURCE


def _first_present_list(payload: dict, *keys: str) -> list[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            return list(value)
    return []


def get_strict_raw_numeric_columns(final_sets: dict, strict_preprocessing_spec: dict) -> list[str]:
    return _first_present_list(final_sets, "strict_raw_numeric_columns") or list(
        strict_preprocessing_spec["strict_numeric_columns"]
    )


def get_strict_base_columns(final_sets: dict, strict_preprocessing_spec: dict) -> list[str]:
    strict_base_columns = _first_present_list(final_sets, "strict_base_columns")
    if strict_base_columns:
        return strict_base_columns

    strict_model_feature_cols = (
        strict_preprocessing_spec["strict_numeric_columns"] + strict_preprocessing_spec["strict_categorical_columns"]
    )
    strict_missing_indicator_feature_cols = [
        f"{column}__missing" for column in strict_preprocessing_spec["numeric_missing_indicator_columns"]
    ]
    return list(dict.fromkeys(strict_model_feature_cols + strict_missing_indicator_feature_cols))


def get_strict_fe_columns(final_sets: dict) -> list[str]:
    return _first_present_list(final_sets, "strict_fe_columns", "selected_engineered_lite_v3_columns")


def get_strict_fe_expanded_columns(final_sets: dict) -> list[str]:
    return _first_present_list(final_sets, "strict_fe_expanded_columns", "selected_engineered_v3_columns")


def get_strict_main_input_columns(final_sets: dict) -> list[str]:
    return _first_present_list(final_sets, "strict_main_input_columns", "strict_final_v3_columns", "strict_lite_final_v3_columns")


def get_relaxed_columns(final_sets: dict) -> list[str]:
    return _first_present_list(final_sets, "relaxed_columns", "relaxed_final_v3_columns")


def apply_strict_preprocessing(input_df, spec: dict):
    import pandas as pd

    output_df = input_df.copy()

    for column in spec.get("drop_columns", []):
        if column in output_df.columns:
            output_df = output_df.drop(columns=column)

    for column in spec.get("numeric_missing_indicator_columns", []):
        if column in output_df.columns:
            output_df[f"{column}__missing"] = output_df[column].isna().astype(int)

    for column, median_value in spec.get("numeric_median_imputation", {}).items():
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce").fillna(median_value)

    for column in spec.get("strict_categorical_columns", []):
        if column in output_df.columns:
            output_df[column] = output_df[column].fillna(spec["categorical_fill_value"])

    for column in spec.get("log1p_candidate_columns", []):
        if column in output_df.columns:
            output_df[column] = np.log1p(output_df[column])

    for column, params in spec.get("robust_scaling_params", {}).items():
        if column in output_df.columns:
            scale = params["scale"] if params["scale"] != 0 else 1.0
            output_df[column] = (output_df[column] - params["center"]) / scale

    return output_df


def circular_abs_diff(a, b):
    diff = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def build_engineered_feature_frame(raw_numeric_df):
    import pandas as pd

    eps = 1e-6
    engineered_df = pd.DataFrame(index=raw_numeric_df.index)

    A = raw_numeric_df["tbp_lv_A"]
    Aext = raw_numeric_df["tbp_lv_Aext"]
    B = raw_numeric_df["tbp_lv_B"]
    Bext = raw_numeric_df["tbp_lv_Bext"]
    C = raw_numeric_df["tbp_lv_C"]
    Cext = raw_numeric_df["tbp_lv_Cext"]
    H = raw_numeric_df["tbp_lv_H"]
    Hext = raw_numeric_df["tbp_lv_Hext"]
    L = raw_numeric_df["tbp_lv_L"]
    Lext = raw_numeric_df["tbp_lv_Lext"]
    area = raw_numeric_df["tbp_lv_areaMM2"]
    area_perim = raw_numeric_df["tbp_lv_area_perim_ratio"]
    color_std = raw_numeric_df["tbp_lv_color_std_mean"]
    deltaA = raw_numeric_df["tbp_lv_deltaA"]
    deltaB = raw_numeric_df["tbp_lv_deltaB"]
    deltaL = raw_numeric_df["tbp_lv_deltaL"]
    deltaLB = raw_numeric_df["tbp_lv_deltaLB"]
    deltaLBnorm = raw_numeric_df["tbp_lv_deltaLBnorm"]
    ecc = raw_numeric_df["tbp_lv_eccentricity"]
    minor = raw_numeric_df["tbp_lv_minorAxisMM"]
    nevi = raw_numeric_df["tbp_lv_nevi_confidence"]
    border = raw_numeric_df["tbp_lv_norm_border"]
    norm_color = raw_numeric_df["tbp_lv_norm_color"]
    perim = raw_numeric_df["tbp_lv_perimeterMM"]
    radial = raw_numeric_df["tbp_lv_radial_color_std_max"]
    stdL = raw_numeric_df["tbp_lv_stdL"]
    stdLExt = raw_numeric_df["tbp_lv_stdLExt"]
    symm = raw_numeric_df["tbp_lv_symm_2axis"]
    x = raw_numeric_df["tbp_lv_x"]
    y = raw_numeric_df["tbp_lv_y"]
    z = raw_numeric_df["tbp_lv_z"]
    age = raw_numeric_df["age_approx"]
    long_diam = raw_numeric_df["clin_size_long_diam_mm"]

    contrast_euclidean = np.sqrt(deltaL**2 + deltaA**2 + deltaB**2)
    hue_gap = circular_abs_diff(H, Hext)
    xyz_radius = np.sqrt(x**2 + y**2 + z**2)
    xz_radius = np.sqrt(x**2 + z**2)

    engineered_df["feat_color_contrast_euclidean"] = contrast_euclidean
    engineered_df["feat_color_contrast_ab"] = np.sqrt(deltaA**2 + deltaB**2)
    engineered_df["feat_color_internal_magnitude"] = np.sqrt(A**2 + B**2 + C**2)
    engineered_df["feat_color_external_magnitude"] = np.sqrt(Aext**2 + Bext**2 + Cext**2)
    engineered_df["feat_color_internal_external_gap"] = np.sqrt((A - Aext) ** 2 + (B - Bext) ** 2 + (L - Lext) ** 2)
    engineered_df["feat_hue_circular_gap"] = hue_gap
    engineered_df["feat_lightness_normalized_gap"] = (L - Lext) / (L + Lext + eps)
    engineered_df["feat_chroma_normalized_gap"] = (C - Cext) / (C + Cext + eps)
    engineered_df["feat_red_green_normalized_gap"] = (A - Aext) / (np.abs(A) + np.abs(Aext) + eps)
    engineered_df["feat_blue_yellow_normalized_gap"] = (B - Bext) / (np.abs(B) + np.abs(Bext) + eps)
    engineered_df["feat_deltaLB_to_stdL"] = deltaLB / (stdL + eps)
    engineered_df["feat_deltaLBnorm_to_stdLExt"] = deltaLBnorm / (stdLExt + eps)
    engineered_df["feat_contrast_to_color_variation"] = contrast_euclidean / (color_std + eps)
    engineered_df["feat_contrast_to_radial_variation"] = contrast_euclidean / (radial + eps)
    engineered_df["feat_color_variation_total"] = color_std + radial + stdL + stdLExt
    engineered_df["feat_internal_external_std_ratio"] = stdL / (stdLExt + eps)
    engineered_df["feat_internal_external_std_balance"] = (stdL - stdLExt) / (stdL + stdLExt + eps)
    engineered_df["feat_hue_color_coupling"] = hue_gap * (norm_color + eps)

    engineered_df["feat_border_color_interaction"] = border * norm_color
    engineered_df["feat_symmetry_border_interaction"] = symm * border
    engineered_df["feat_symmetry_color_interaction"] = symm * norm_color
    engineered_df["feat_symmetry_contrast_interaction"] = symm * contrast_euclidean
    engineered_df["feat_border_contrast_interaction"] = border * contrast_euclidean
    engineered_df["feat_border_radial_color_interaction"] = border * radial
    engineered_df["feat_border_colorstd_interaction"] = border * color_std
    engineered_df["feat_symmetry_radial_color_interaction"] = symm * radial
    engineered_df["feat_symmetry_colorstd_interaction"] = symm * color_std
    engineered_df["feat_architecture_proxy_sum"] = symm + border + norm_color
    engineered_df["feat_architecture_proxy_product"] = (symm + eps) * (border + eps) * (norm_color + eps)
    engineered_df["feat_cash_proxy_raw"] = symm + border + norm_color + color_std
    engineered_df["feat_abcd_proxy_raw"] = 1.3 * symm + 0.1 * border + 0.5 * norm_color + 0.5 * contrast_euclidean
    engineered_df["feat_homogeneity_inverse"] = 1.0 / (1.0 + color_std + radial)
    engineered_df["feat_border_to_color_ratio"] = border / (norm_color + eps)
    engineered_df["feat_color_to_border_ratio"] = norm_color / (border + eps)
    engineered_df["feat_radial_to_global_color_ratio"] = radial / (color_std + eps)
    engineered_df["feat_structure_dispersion_proxy"] = norm_color + color_std + radial

    engineered_df["feat_long_to_minor_ratio"] = long_diam / (minor + eps)
    engineered_df["feat_minor_to_long_ratio"] = minor / (long_diam + eps)
    engineered_df["feat_perimeter_to_long_ratio"] = perim / (long_diam + eps)
    engineered_df["feat_perimeter_to_minor_ratio"] = perim / (minor + eps)
    engineered_df["feat_area_to_long_sq"] = area / (long_diam**2 + eps)
    engineered_df["feat_area_to_perimeter_sq"] = area / (perim**2 + eps)
    engineered_df["feat_perimeter_sq_to_area"] = perim**2 / (area + eps)
    engineered_df["feat_ellipse_fill_ratio"] = area / (long_diam * minor + eps)
    engineered_df["feat_area_perim_border_coupling"] = area_perim * border
    engineered_df["feat_area_perim_symmetry_coupling"] = area_perim * symm
    engineered_df["feat_area_eccentricity_coupling"] = area * ecc
    engineered_df["feat_border_eccentricity_coupling"] = border * ecc
    engineered_df["feat_symmetry_eccentricity_coupling"] = symm * ecc
    engineered_df["feat_diameter_border_coupling"] = long_diam * border
    engineered_df["feat_diameter_color_coupling"] = long_diam * norm_color
    engineered_df["feat_diameter_symmetry_coupling"] = long_diam * symm
    engineered_df["feat_size_shape_proxy"] = long_diam * perim / (area + eps)
    engineered_df["feat_compactness_eccentricity"] = (perim**2 / (area + eps)) * (ecc + eps)
    engineered_df["feat_long_minor_difference"] = long_diam - minor
    engineered_df["feat_perimeter_area_balance"] = perim / np.sqrt(area + eps)

    engineered_df["feat_age_size_interaction"] = age * long_diam
    engineered_df["feat_age_area_interaction"] = age * area
    engineered_df["feat_age_perimeter_interaction"] = age * perim
    engineered_df["feat_age_symmetry_interaction"] = age * symm
    engineered_df["feat_age_contrast_interaction"] = age * contrast_euclidean
    engineered_df["feat_age_color_interaction"] = age * norm_color
    engineered_df["feat_nevi_border_interaction"] = nevi * border
    engineered_df["feat_nevi_color_interaction"] = nevi * norm_color
    engineered_df["feat_nevi_symmetry_interaction"] = nevi * symm
    engineered_df["feat_nevi_contrast_interaction"] = nevi * contrast_euclidean

    engineered_df["feat_xyz_radius"] = xyz_radius
    engineered_df["feat_xy_radius"] = np.sqrt(x**2 + y**2)
    engineered_df["feat_yz_radius"] = np.sqrt(y**2 + z**2)
    engineered_df["feat_xz_radius"] = xz_radius
    engineered_df["feat_abs_y"] = np.abs(y)
    engineered_df["feat_abs_z"] = np.abs(z)
    engineered_df["feat_area_to_xyz_radius"] = area / (xyz_radius + eps)
    engineered_df["feat_long_to_xyz_radius"] = long_diam / (xyz_radius + eps)
    engineered_df["feat_symmetry_to_xyz_radius"] = symm / (xyz_radius + eps)
    engineered_df["feat_vertical_size_interaction"] = np.abs(y) * long_diam

    return engineered_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_engineered_features_for_input(
    input_df,
    *,
    strict_preprocessing_spec: dict,
    selected_engineered_cols: list[str],
    engineered_preprocess_spec: dict,
):
    import pandas as pd

    strict_numeric_model_cols = strict_preprocessing_spec["strict_numeric_columns"]
    raw_numeric_df = input_df[strict_numeric_model_cols].copy()
    for column, median_value in strict_preprocessing_spec["numeric_median_imputation"].items():
        if column in raw_numeric_df.columns:
            raw_numeric_df[column] = pd.to_numeric(raw_numeric_df[column], errors="coerce").fillna(median_value)

    engineered_all_df = build_engineered_feature_frame(raw_numeric_df)
    engineered_selected_df = engineered_all_df[selected_engineered_cols].copy()

    for column in engineered_preprocess_spec.get("log1p_candidate_columns", []):
        if column in engineered_selected_df.columns:
            engineered_selected_df[column] = np.log1p(engineered_selected_df[column])

    for column, params in engineered_preprocess_spec.get("robust_scaling_params", {}).items():
        if column in engineered_selected_df.columns:
            scale = params["scale"] if params["scale"] != 0 else 1.0
            engineered_selected_df[column] = (engineered_selected_df[column] - params["center"]) / scale

    return engineered_selected_df


def build_final_feature_frame(
    input_df,
    *,
    strict_preprocessing_spec: dict,
    strict_base_columns: list[str] | None = None,
    selected_engineered_cols: list[str],
    engineered_preprocess_spec: dict,
    relaxed_extra_cols: list[str] | None = None,
):
    if strict_base_columns is None:
        strict_base_columns = get_strict_base_columns({}, strict_preprocessing_spec)

    strict_processed_df = apply_strict_preprocessing(input_df, strict_preprocessing_spec)
    final_df = strict_processed_df[strict_base_columns].copy()
    engineered_df = build_engineered_features_for_input(
        input_df,
        strict_preprocessing_spec=strict_preprocessing_spec,
        selected_engineered_cols=selected_engineered_cols,
        engineered_preprocess_spec=engineered_preprocess_spec,
    )
    final_df = final_df.join(engineered_df)

    if relaxed_extra_cols:
        relaxed_df = input_df[relaxed_extra_cols].copy()
        for column in relaxed_extra_cols:
            if column in relaxed_df.columns:
                relaxed_df[column] = relaxed_df[column].fillna("Missing")
        final_df = final_df.join(relaxed_df)

    return final_df


def build_named_final_feature_frame(input_df, eda_dir: str | Path, feature_set_name: str):
    final_sets = load_final_feature_sets_v3(eda_dir)
    strict_preprocessing_spec = load_strict_preprocessing_spec(eda_dir)
    normalized_feature_set_name = normalize_feature_set_name(feature_set_name)
    strict_base_columns = get_strict_base_columns(final_sets, strict_preprocessing_spec)

    if normalized_feature_set_name == LEGACY_STRICT_FULL:
        selected_engineered_cols = get_strict_fe_expanded_columns(final_sets)
        engineered_preprocess_spec = final_sets.get("strict_fe_expanded_preprocess_spec", final_sets["engineered_preprocess_spec_v3"])
        relaxed_extra_cols = None
        expected_columns = _first_present_list(final_sets, "strict_expanded_input_columns", "strict_full_v3_columns")
    elif normalized_feature_set_name == STRICT_BASE:
        selected_engineered_cols = []
        engineered_preprocess_spec = {"log1p_candidate_columns": [], "robust_scaling_params": {}}
        relaxed_extra_cols = None
        expected_columns = get_strict_base_columns(final_sets, strict_preprocessing_spec)
    elif normalized_feature_set_name == STRICT_FE:
        selected_engineered_cols = get_strict_fe_columns(final_sets)
        engineered_preprocess_spec = final_sets.get("strict_fe_preprocess_spec", final_sets["engineered_preprocess_spec_lite_v3"])
        relaxed_extra_cols = None
        expected_columns = get_strict_fe_columns(final_sets)
    elif normalized_feature_set_name == STRICT_MAIN_INPUT:
        selected_engineered_cols = get_strict_fe_columns(final_sets)
        engineered_preprocess_spec = final_sets.get("strict_fe_preprocess_spec", final_sets["engineered_preprocess_spec_lite_v3"])
        relaxed_extra_cols = None
        expected_columns = get_strict_main_input_columns(final_sets)
    elif normalized_feature_set_name == RELAXED:
        selected_engineered_cols = get_strict_fe_columns(final_sets)
        engineered_preprocess_spec = final_sets.get("strict_fe_preprocess_spec", final_sets["engineered_preprocess_spec_lite_v3"])
        relaxed_extra_cols = list(final_sets.get("relaxed_extra_columns", []))
        expected_columns = get_relaxed_columns(final_sets)
    else:
        raise ValueError(f"Unsupported final-input feature set: {feature_set_name}")

    final_df = build_final_feature_frame(
        input_df,
        strict_preprocessing_spec=strict_preprocessing_spec,
        strict_base_columns=strict_base_columns,
        selected_engineered_cols=selected_engineered_cols,
        engineered_preprocess_spec=engineered_preprocess_spec,
        relaxed_extra_cols=relaxed_extra_cols,
    )
    missing_columns = [column for column in expected_columns if column not in final_df.columns]
    if missing_columns:
        raise RuntimeError(
            f"Final tabular frame for '{normalized_feature_set_name}' is missing expected columns: {missing_columns[:10]}"
        )

    return final_df[expected_columns].copy()


def build_final_feature_frames(input_df, eda_dir: str | Path, feature_set_names: list[str]) -> dict[str, object]:
    return {
        normalize_feature_set_name(feature_set_name): build_named_final_feature_frame(input_df, eda_dir, feature_set_name)
        for feature_set_name in feature_set_names
    }
