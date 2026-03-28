from __future__ import annotations

import csv
import json
from pathlib import Path


TARGET_COLUMN = "malignant"
IDENTIFIER_COLUMNS = {"isic_id", "image_path", "image_exists"}
DIAGNOSIS_COLUMNS = {"iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "iddx_full"}


def load_dataset_overview(eda_dir: str | Path) -> dict:
    return json.loads((Path(eda_dir) / "dataset_overview.json").read_text(encoding="utf-8"))


def load_missingness_summary(eda_dir: str | Path) -> dict[str, dict[str, float]]:
    path = Path(eda_dir) / "missingness_summary.csv"
    summary: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            summary[row["column"]] = {
                "missing_ratio": float(row["missing_ratio"]),
                "non_missing_ratio": float(row["non_missing_ratio"]),
            }
    return summary


def load_target_rate_summary(eda_dir: str | Path, column: str) -> dict[str, dict[str, float]]:
    path = Path(eda_dir) / f"target_rate_by_{column}.csv"
    summary: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = row[column]
            summary[key] = {
                "count": int(row["count"]),
                "positive_count": int(row["positive_count"]),
                "negative_count": int(row["negative_count"]),
                "positive_ratio": float(row["positive_ratio"]),
            }
    return summary


def recommend_feature_sets(eda_dir: str | Path) -> dict[str, object]:
    overview = load_dataset_overview(eda_dir)
    missingness = load_missingness_summary(eda_dir)
    iddx_1_rates = load_target_rate_summary(eda_dir, "iddx_1")
    iddx_full_rates = load_target_rate_summary(eda_dir, "iddx_full")

    all_columns = [column for column in overview["columns"] if column != TARGET_COLUMN]
    high_leakage = set(DIAGNOSIS_COLUMNS)

    # EDA 기준 추가 leakage 후보:
    # - mel_thick_mm는 현재 유효값이 모두 양성에만 존재함
    # - mel_mitotic_index는 유효값이 거의 없고 실제 학습 feature로 쓰기 어려움
    if "mel_thick_mm" in missingness:
        high_leakage.add("mel_thick_mm")
    if "mel_mitotic_index" in missingness:
        high_leakage.add("mel_mitotic_index")

    strict = []
    relaxed = []
    oracle = []
    excluded = []

    for column in all_columns:
        if column in IDENTIFIER_COLUMNS:
            excluded.append(column)
            continue

        oracle.append(column)
        if column in high_leakage:
            continue

        # 결측이 과도하면 strict에서는 제외하고 relaxed에서만 검토한다.
        non_missing_ratio = missingness.get(column, {}).get("non_missing_ratio", 0.0)
        if non_missing_ratio >= 0.95:
            strict.append(column)
            relaxed.append(column)
        elif non_missing_ratio >= 0.05:
            relaxed.append(column)

    rationales = {
        "strict": [
            "식별자와 진단 계열 컬럼을 제외합니다.",
            "결측이 매우 심한 컬럼을 제외합니다.",
            "초기 현실형 baseline 비교 기준으로 사용합니다.",
        ],
        "relaxed": [
            "strict 세트에 더해, 결측이 많지만 검토 가치가 있는 컬럼을 포함합니다.",
            "운영 가능성보다 탐색적 성능 비교에 가깝습니다.",
        ],
        "oracle": [
            "진단 계열 컬럼까지 포함한 상한선 성격의 세트입니다.",
            "현실적인 baseline보다는 leakage 영향 확인용입니다.",
        ],
    }

    evidence = {
        "iddx_1_target_rates": iddx_1_rates,
        "iddx_full_target_rates_sample": dict(list(iddx_full_rates.items())[:10]),
        "missingness_snapshot": {
            key: missingness[key]
            for key in sorted(missingness)
            if key in {"lesion_id", "mel_mitotic_index", "mel_thick_mm", "tbp_lv_dnn_lesion_confidence"}
        },
    }

    return {
        "target_column": TARGET_COLUMN,
        "excluded_columns": sorted(excluded),
        "high_leakage_risk_columns": sorted(high_leakage),
        "feature_sets": {
            "strict": strict,
            "relaxed": relaxed,
            "oracle": oracle,
        },
        "rationales": rationales,
        "evidence": evidence,
    }
