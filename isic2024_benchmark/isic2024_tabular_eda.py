from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from isic2024_benchmark.runtime_env import ensure_expected_conda_env
from isic2024_benchmark.tabular_data import DEFAULT_DATASET_ROOT, DEFAULT_TARGET_COLUMN, load_tabular_dataframe


TARGET_RATE_COLUMNS = ("iddx_1", "iddx_full", "attribution", "copyright_license")
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
CAUTION_COLUMNS = {
    "isic_id",
    "patient_id",
    "lesion_id",
    "image_path",
    "image_exists",
    "split_group_id",
    "attribution",
    "copyright_license",
}
CATEGORICAL_SUMMARY_EXCLUSIONS = {"image_path", "split_group_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ISIC2024 challenge tabular EDA artifacts.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default="artifacts/eda/isic2024")
    parser.add_argument("--top-k", type=int, default=20, help="Rows to keep for top-category summaries.")
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_tabular_dataframe(args.dataset_root, include_image_columns=True)
    target_column = DEFAULT_TARGET_COLUMN
    ordered_columns = sorted(frame.columns.tolist())
    numeric_columns = [
        column
        for column in ordered_columns
        if column not in {target_column, "image_exists"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column
        for column in ordered_columns
        if column not in {target_column, *CATEGORICAL_SUMMARY_EXCLUSIONS}
        and column not in numeric_columns
    ]

    preview_rows = frame.head(20).copy()
    preview_rows = preview_rows[[column for column in ordered_columns if column in preview_rows.columns]]
    preview_rows.to_csv(output_dir / "preview_rows.csv", index=False)

    overview = build_dataset_overview(
        frame,
        dataset_root=args.dataset_root,
        ordered_columns=ordered_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    write_json(output_dir / "dataset_overview.json", overview)
    write_missingness_csv(frame, ordered_columns, output_dir / "missingness_summary.csv")
    write_numeric_summary_csv(frame, numeric_columns, output_dir / "numeric_summary.csv")
    write_categorical_summary_csv(frame, categorical_columns, output_dir / "categorical_summary.csv", top_k=args.top_k)
    for column in TARGET_RATE_COLUMNS:
        if column in frame.columns:
            write_target_rate_csv(frame, target_column, column, output_dir / f"target_rate_by_{column}.csv", top_k=args.top_k)
    write_json(output_dir / "feature_groups.json", classify_feature_groups(ordered_columns))
    write_report_markdown(
        output_dir / "report.md",
        overview=overview,
        missingness=pd.read_csv(output_dir / "missingness_summary.csv"),
        numeric_columns=numeric_columns,
        leakage_groups=classify_feature_groups(ordered_columns),
    )
    print(f"Saved ISIC2024 challenge tabular EDA artifacts to {output_dir}")


def build_dataset_overview(
    frame: pd.DataFrame,
    *,
    dataset_root: str | Path,
    ordered_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, object]:
    target_distribution = frame[DEFAULT_TARGET_COLUMN].value_counts(dropna=False).to_dict()
    image_exists = frame["image_exists"].value_counts(dropna=False).to_dict()
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "rows": int(len(frame)),
        "target_column": DEFAULT_TARGET_COLUMN,
        "target_distribution": {
            "negative": int(target_distribution.get(0, 0)),
            "positive": int(target_distribution.get(1, 0)),
            "positive_ratio": float(frame[DEFAULT_TARGET_COLUMN].mean()),
        },
        "image_alignment": {
            "image_exists": int(image_exists.get(1, 0)),
            "image_missing": int(image_exists.get(0, 0)),
        },
        "column_count": len(ordered_columns),
        "columns": ordered_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "group_split_policy": "patient_id -> lesion_id -> isic_id",
    }


def write_missingness_csv(frame: pd.DataFrame, ordered_columns: list[str], output_path: Path) -> None:
    rows = []
    row_count = max(len(frame), 1)
    for column in ordered_columns:
        non_missing_mask = get_non_missing_mask(frame[column])
        non_missing_count = int(non_missing_mask.sum())
        missing_count = int(len(frame) - non_missing_count)
        rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_ratio": f"{missing_count / row_count:.6f}",
                "non_missing_count": non_missing_count,
                "non_missing_ratio": f"{non_missing_count / row_count:.6f}",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_numeric_summary_csv(frame: pd.DataFrame, numeric_columns: list[str], output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for column in numeric_columns:
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(build_numeric_summary_row(column, "all", numeric_series))
        for target_value in (0, 1):
            rows.append(
                build_numeric_summary_row(
                    column,
                    f"target_{target_value}",
                    numeric_series[frame[DEFAULT_TARGET_COLUMN] == target_value],
                )
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_numeric_summary_row(column: str, group: str, series: pd.Series) -> dict[str, object]:
    values = series.dropna()
    if values.empty:
        return {"column": column, "group": group, "count": 0, "mean": "", "median": "", "min": "", "max": ""}
    return {
        "column": column,
        "group": group,
        "count": int(values.shape[0]),
        "mean": f"{values.mean():.6f}",
        "median": f"{values.median():.6f}",
        "min": f"{values.min():.6f}",
        "max": f"{values.max():.6f}",
    }


def write_categorical_summary_csv(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    output_path: Path,
    *,
    top_k: int,
) -> None:
    rows: list[dict[str, object]] = []
    row_count = max(len(frame), 1)
    for column in categorical_columns:
        normalized = normalize_category_series(frame[column])
        normalized = normalized[normalized != ""]
        if normalized.empty:
            continue
        counts = normalized.value_counts().head(top_k)
        unique_count = int(normalized.nunique())
        for value, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "unique_count": unique_count,
                    "value": value,
                    "count": int(count),
                    "ratio": f"{count / row_count:.6f}",
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_target_rate_csv(
    frame: pd.DataFrame,
    target_column: str,
    column: str,
    output_path: Path,
    *,
    top_k: int,
) -> None:
    normalized = normalize_category_series(frame[column])
    valid_mask = normalized != ""
    working = pd.DataFrame({column: normalized[valid_mask], target_column: frame.loc[valid_mask, target_column].astype(int)})
    if working.empty:
        pd.DataFrame(columns=[column, "count", "positive_count", "negative_count", "positive_ratio"]).to_csv(output_path, index=False)
        return

    summary = (
        working.groupby(column)[target_column]
        .agg(count="size", positive_count="sum")
        .sort_values("count", ascending=False)
        .head(top_k)
        .reset_index()
    )
    summary["negative_count"] = summary["count"] - summary["positive_count"]
    summary["positive_ratio"] = summary["positive_count"] / summary["count"]
    summary.to_csv(output_path, index=False)


def classify_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    safe_columns = []
    caution_columns = []
    high_risk_columns = []

    for column in columns:
        if column == DEFAULT_TARGET_COLUMN:
            caution_columns.append(column)
        elif column in HIGH_LEAKAGE_COLUMNS:
            high_risk_columns.append(column)
        elif column in CAUTION_COLUMNS:
            caution_columns.append(column)
        else:
            safe_columns.append(column)

    return {
        "safe_for_initial_baseline_review": safe_columns,
        "caution_review_needed": caution_columns,
        "high_leakage_risk": high_risk_columns,
    }


def write_report_markdown(
    path: Path,
    *,
    overview: dict[str, object],
    missingness: pd.DataFrame,
    numeric_columns: list[str],
    leakage_groups: dict[str, list[str]],
) -> None:
    top_missing = missingness.sort_values("missing_count", ascending=False).head(10)
    distribution = overview["target_distribution"]
    lines = [
        "# ISIC2024 Challenge Tabular EDA Report",
        "",
        "## 데이터 개요",
        f"- 전체 행 수: `{overview['rows']}`",
        f"- 양성 수: `{distribution['positive']}`",
        f"- 음성 수: `{distribution['negative']}`",
        f"- 양성 비율: `{distribution['positive_ratio']:.6f}`",
        f"- 총 컬럼 수: `{overview['column_count']}`",
        f"- split 정책: `{overview['group_split_policy']}`",
        "",
        "## 결측 상위 컬럼",
    ]
    for row in top_missing.itertuples():
        lines.append(f"- `{row.column}`: `{row.missing_count}`")

    lines.extend(
        [
            "",
            "## 수치형 컬럼",
            f"- 수치형 컬럼 수: `{len(numeric_columns)}`",
            "",
            "## feature 그룹 분류 초안",
            "- 이 분류는 feature set 정책을 설명하기 위한 EDA 초안입니다.",
            f"- 초기 baseline 검토 가능: `{', '.join(leakage_groups['safe_for_initial_baseline_review'])}`",
            f"- 주의 검토 필요: `{', '.join(leakage_groups['caution_review_needed'])}`",
            f"- leakage 위험 높음: `{', '.join(leakage_groups['high_leakage_risk'])}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_non_missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna()
    normalized = normalize_category_series(series)
    return normalized != ""


def normalize_category_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": "", "<na>": ""})
    )


if __name__ == "__main__":
    main()
