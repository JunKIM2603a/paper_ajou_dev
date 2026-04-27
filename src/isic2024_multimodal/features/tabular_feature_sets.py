from __future__ import annotations

import json
from pathlib import Path

from isic2024_multimodal.features.final_tabular_inputs import load_strict_preprocessing_spec
from isic2024_multimodal.data.tabular_dataset import DEFAULT_TARGET_COLUMN
from isic2024_multimodal.features.tabular_terms import (
    FEATURE_SET_ALIASES,
    RELAXED,
    STRICT_BASE,
    STRICT_FE,
    STRICT_MAIN_INPUT,
)


TARGET_COLUMN = DEFAULT_TARGET_COLUMN
FINAL_INPUTS_RELATIVE_PATH = Path("final_inputs") / "final_feature_sets_v3.json"


def load_final_feature_sets_v3(eda_dir: str | Path) -> dict:
    path = Path(eda_dir) / FINAL_INPUTS_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Notebook-derived final feature payload was not found at "
            f"'{path}'. Run 'notebooks/isic_2024/isic2024_eda_20260411.ipynb' and regenerate "
            "the final_inputs cells before launching tabular baselines."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def recommend_feature_sets(eda_dir: str | Path) -> dict[str, object]:
    return recommend_feature_sets_from_final_inputs(eda_dir)


def recommend_feature_sets_from_final_inputs(eda_dir: str | Path) -> dict[str, object]:
    final_sets = load_final_feature_sets_v3(eda_dir)
    strict_preprocessing_spec = load_strict_preprocessing_spec(eda_dir)

    strict_raw_numeric = list(final_sets.get("strict_raw_numeric_columns", strict_preprocessing_spec["strict_numeric_columns"]))
    strict_base = list(
        final_sets.get(
            "strict_base_columns",
            list(
                dict.fromkeys(
                    strict_preprocessing_spec["strict_numeric_columns"]
                    + strict_preprocessing_spec["strict_categorical_columns"]
                    + [f"{column}__missing" for column in strict_preprocessing_spec["numeric_missing_indicator_columns"]]
                )
            ),
        )
    )
    strict_fe = list(final_sets.get("strict_fe_columns", final_sets.get("selected_engineered_lite_v3_columns", [])))
    strict_main_input = list(final_sets.get("strict_main_input_columns", final_sets.get("strict_final_v3_columns", [])))
    relaxed = list(final_sets.get("relaxed_columns", final_sets.get("relaxed_final_v3_columns", [])))
    selected_engineered = list(final_sets.get("strict_fe_expanded_columns", final_sets.get("selected_engineered_v3_columns", [])))
    oracle_source = list(final_sets.get("oracle_supervision_source_columns", []))
    reference_only = list(final_sets.get("reference_only_columns", []))
    label_source = list(final_sets.get("label_source_columns", []))

    evidence = {
        "feature_sets_source": FINAL_INPUTS_RELATIVE_PATH.as_posix(),
        "notebook_source": "notebooks/isic_2024/isic2024_eda_20260411.ipynb",
        "strict_raw_numeric_columns": strict_raw_numeric,
        "strict_base_columns": strict_base,
        "strict_fe_columns": strict_fe,
        "strict_fe_expanded_columns": selected_engineered,
        "oracle_supervision_source_columns": oracle_source,
        "reference_only_columns": reference_only,
        "label_source_columns": label_source,
    }

    return {
        "target_column": TARGET_COLUMN,
        "strict_min_non_missing_ratio": None,
        "relaxed_min_non_missing_ratio": None,
        "excluded_columns": [],
        "high_leakage_risk_columns": sorted(set(reference_only + label_source + oracle_source)),
        "feature_sets": {
            STRICT_BASE: strict_base,
            STRICT_FE: strict_fe,
            STRICT_MAIN_INPUT: strict_main_input,
            RELAXED: relaxed,
        },
        "feature_set_aliases": FEATURE_SET_ALIASES,
        "rationales": {
            STRICT_BASE: [
                "notebook final_inputs의 strict_base_columns를 사용합니다.",
                "전처리된 base metadata만으로 구성된 tabular 기준선입니다.",
            ],
            STRICT_FE: [
                "notebook final_inputs의 strict_fe_columns를 사용합니다.",
                "최종 선택 engineered feature만으로 구성된 FE-only 기준선입니다.",
            ],
            STRICT_MAIN_INPUT: [
                "notebook final_inputs의 strict_main_input_columns를 사용합니다.",
                "현재 논문 본선 기준의 메인 tabular 입력 세트입니다.",
            ],
            RELAXED: [
                "notebook final_inputs의 relaxed_columns를 사용합니다.",
                f"{STRICT_MAIN_INPUT}에 provenance/context 컬럼을 일부 추가한 보조 비교 세트입니다.",
            ],
        },
        "evidence": evidence,
    }
