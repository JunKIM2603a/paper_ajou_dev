from __future__ import annotations

import csv
import json
from pathlib import Path

from isic2024_benchmark.tabular_data import DEFAULT_TARGET_COLUMN


TARGET_COLUMN = DEFAULT_TARGET_COLUMN
IDENTIFIER_COLUMNS = {"isic_id", "patient_id", "image_path", "image_exists", "split_group_id"}
RELAXED_ONLY_COLUMNS = {"lesion_id", "attribution", "copyright_license"}
HIGH_LEAKAGE_COLUMNS = {
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "iddx_full",
    "mel_mitotic_index",
    "mel_thick_mm",
}
STRICT_MIN_NON_MISSING_RATIO = 0.95
RELAXED_MIN_NON_MISSING_RATIO = 0.05


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
    if not path.exists():
        return {}
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
    strict: list[str] = []
    relaxed: list[str] = []
    oracle: list[str] = []

    for column in all_columns:
        non_missing_ratio = missingness.get(column, {}).get("non_missing_ratio", 0.0)

        if column in IDENTIFIER_COLUMNS:
            continue
        if column in RELAXED_ONLY_COLUMNS:
            relaxed.append(column)
            oracle.append(column)
            continue
        if column in HIGH_LEAKAGE_COLUMNS:
            oracle.append(column)
            continue

        if non_missing_ratio >= STRICT_MIN_NON_MISSING_RATIO:
            strict.append(column)
            relaxed.append(column)
            oracle.append(column)
        elif non_missing_ratio >= RELAXED_MIN_NON_MISSING_RATIO:
            relaxed.append(column)
            oracle.append(column)

    included_columns = set(strict) | set(relaxed) | set(oracle)
    excluded_columns = sorted(column for column in all_columns if column not in included_columns)

    evidence = {
        "iddx_1_target_rates": iddx_1_rates,
        "iddx_full_target_rates_sample": dict(list(iddx_full_rates.items())[:10]),
        "missingness_snapshot": {
            key: missingness[key]
            for key in sorted(missingness)
            if key in {"lesion_id", "mel_mitotic_index", "mel_thick_mm", "tbp_lv_dnn_lesion_confidence", "patient_id"}
        },
    }

    return {
        "target_column": TARGET_COLUMN,
        "strict_min_non_missing_ratio": STRICT_MIN_NON_MISSING_RATIO,
        "relaxed_min_non_missing_ratio": RELAXED_MIN_NON_MISSING_RATIO,
        "excluded_columns": excluded_columns,
        "high_leakage_risk_columns": sorted(HIGH_LEAKAGE_COLUMNS),
        "feature_sets": {
            "strict": strict,
            "relaxed": relaxed,
            "oracle": oracle,
        },
        "rationales": {
            "strict": [
                "TBP/기본 임상 메타데이터 중심의 현실형 baseline 세트입니다.",
                "식별자, 환자/병변 식별 정보, 진단 계열 컬럼, mel 계열 컬럼, 기관/라이선스 컬럼을 제외합니다.",
                f"결측이 심한 컬럼은 non-missing ratio `{STRICT_MIN_NON_MISSING_RATIO:.2f}` 이상일 때만 포함합니다.",
            ],
            "relaxed": [
                "strict 세트에 더해 lesion/기관 메타데이터와 중간 결측 컬럼을 포함합니다.",
                "메인 결과표보다는 보조 비교와 편향 점검에 적합합니다.",
            ],
            "oracle": [
                "relaxed 세트에 진단 계열과 mel 계열 컬럼을 추가한 leakage 상한선 세트입니다.",
                "현실형 baseline이 아니라 leakage 영향을 확인하기 위한 참고용 세트입니다.",
            ],
        },
        "evidence": evidence,
    }
