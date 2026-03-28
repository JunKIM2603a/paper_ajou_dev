from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from isic2024_benchmark.tabular_data import iter_merged_tabular_rows, resolve_isic2024_dataset_root


TARGET_COLUMN = "malignant"
NUMERIC_CANDIDATES = {"tbp_lv_dnn_lesion_confidence", "mel_mitotic_index", "mel_thick_mm"}
TARGET_RATE_COLUMNS = ("iddx_1", "iddx_full", "attribution", "copyright_license")
LEAKAGE_RISK_COLUMNS = ("iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "iddx_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ISIC2024 tabular EDA artifacts.")
    parser.add_argument("--dataset-root", default="dataset/ISIC2024")
    parser.add_argument("--output-dir", default="artifacts/eda/isic2024")
    parser.add_argument("--top-k", type=int, default=20, help="Rows to keep for top-category summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_isic2024_dataset_root(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_count = 0
    target_counter: Counter[int] = Counter()
    image_exists_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    non_missing_counter: Counter[str] = Counter()
    categorical_counters: dict[str, Counter[str]] = defaultdict(Counter)
    target_rate_counters: dict[str, dict[str, Counter[int]]] = {
        column: defaultdict(Counter) for column in TARGET_RATE_COLUMNS
    }
    numeric_values: dict[str, list[float]] = defaultdict(list)
    numeric_values_by_target: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    preview_rows: list[dict[str, str]] = []
    all_columns: set[str] = set()

    for row in iter_merged_tabular_rows(paths):
        row_count += 1
        all_columns.update(row.keys())
        target = parse_binary_label(row.get(TARGET_COLUMN, ""))
        target_counter[target] += 1
        image_exists_counter[row.get("image_exists", "0")] += 1

        if len(preview_rows) < 20:
            preview_rows.append(row)

        for column, value in row.items():
            normalized = value.strip() if isinstance(value, str) else str(value)
            if normalized == "":
                missing_counter[column] += 1
                continue

            non_missing_counter[column] += 1

            if column in NUMERIC_CANDIDATES:
                numeric_value = safe_float(normalized)
                if numeric_value is not None and not math.isnan(numeric_value):
                    numeric_values[column].append(numeric_value)
                    numeric_values_by_target[column][target].append(numeric_value)
                continue

            categorical_counters[column][normalized] += 1
            if column in target_rate_counters:
                target_rate_counters[column][normalized][target] += 1

    ordered_columns = sorted(all_columns)
    leakage_groups = classify_feature_groups(ordered_columns)

    write_json(
        output_dir / "dataset_overview.json",
        {
            "dataset_root": str(paths.dataset_root.resolve()),
            "rows": row_count,
            "target_column": TARGET_COLUMN,
            "target_distribution": {
                "negative": target_counter.get(0, 0),
                "positive": target_counter.get(1, 0),
                "positive_ratio": safe_divide(target_counter.get(1, 0), row_count),
            },
            "image_alignment": {
                "image_exists": image_exists_counter.get("1", 0),
                "image_missing": image_exists_counter.get("0", 0),
            },
            "column_count": len(ordered_columns),
            "columns": ordered_columns,
        },
    )
    write_csv(output_dir / "preview_rows.csv", preview_rows, ordered_columns)
    write_missingness_csv(output_dir / "missingness_summary.csv", ordered_columns, row_count, missing_counter, non_missing_counter)
    write_numeric_summary_csv(output_dir / "numeric_summary.csv", numeric_values, numeric_values_by_target)
    write_categorical_summary_csv(
        output_dir / "categorical_summary.csv",
        categorical_counters,
        row_count=row_count,
        top_k=args.top_k,
    )
    for column, counters in target_rate_counters.items():
        write_target_rate_csv(output_dir / f"target_rate_by_{column}.csv", column, counters, args.top_k)
    write_json(output_dir / "feature_groups.json", leakage_groups)
    write_report_markdown(
        output_dir / "report.md",
        row_count=row_count,
        target_counter=target_counter,
        image_exists_counter=image_exists_counter,
        ordered_columns=ordered_columns,
        missing_counter=missing_counter,
        numeric_values=numeric_values,
        leakage_groups=leakage_groups,
    )
    print(f"Saved ISIC2024 tabular EDA artifacts to {output_dir}")


def parse_binary_label(value: str) -> int:
    normalized = value.strip()
    if normalized in {"1", "1.0"}:
        return 1
    return 0


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def classify_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    safe_columns: list[str] = []
    caution_columns: list[str] = []
    high_risk_columns: list[str] = []

    for column in columns:
        if column in {"isic_id", "image_path", "image_exists", TARGET_COLUMN}:
            caution_columns.append(column)
        elif column in LEAKAGE_RISK_COLUMNS:
            high_risk_columns.append(column)
        elif column in {"lesion_id", "attribution", "copyright_license"}:
            caution_columns.append(column)
        else:
            safe_columns.append(column)

    return {
        "safe_for_initial_baseline_review": safe_columns,
        "caution_review_needed": caution_columns,
        "high_leakage_risk": high_risk_columns,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_missingness_csv(
    path: Path,
    ordered_columns: list[str],
    row_count: int,
    missing_counter: Counter[str],
    non_missing_counter: Counter[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["column", "missing_count", "missing_ratio", "non_missing_count", "non_missing_ratio"])
        for column in ordered_columns:
            missing_count = missing_counter.get(column, 0)
            non_missing_count = non_missing_counter.get(column, 0)
            writer.writerow(
                [
                    column,
                    missing_count,
                    f"{safe_divide(missing_count, row_count):.6f}",
                    non_missing_count,
                    f"{safe_divide(non_missing_count, row_count):.6f}",
                ]
            )


def summarize_numeric(values: list[float]) -> dict[str, str]:
    if not values:
        return {
            "count": "0",
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
        }
    return {
        "count": str(len(values)),
        "mean": f"{mean(values):.6f}",
        "median": f"{median(values):.6f}",
        "min": f"{min(values):.6f}",
        "max": f"{max(values):.6f}",
    }


def write_numeric_summary_csv(
    path: Path,
    numeric_values: dict[str, list[float]],
    numeric_values_by_target: dict[str, dict[int, list[float]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["column", "group", "count", "mean", "median", "min", "max"])
        for column in sorted(NUMERIC_CANDIDATES):
            overall = summarize_numeric(numeric_values.get(column, []))
            writer.writerow([column, "all", overall["count"], overall["mean"], overall["median"], overall["min"], overall["max"]])
            for target in [0, 1]:
                stats = summarize_numeric(numeric_values_by_target.get(column, {}).get(target, []))
                writer.writerow(
                    [column, f"target_{target}", stats["count"], stats["mean"], stats["median"], stats["min"], stats["max"]]
                )


def write_categorical_summary_csv(
    path: Path,
    categorical_counters: dict[str, Counter[str]],
    *,
    row_count: int,
    top_k: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["column", "unique_count", "value", "count", "ratio"])
        for column in sorted(categorical_counters):
            counter = categorical_counters[column]
            unique_count = len(counter)
            for value, count in counter.most_common(top_k):
                writer.writerow([column, unique_count, value, count, f"{safe_divide(count, row_count):.6f}"])


def write_target_rate_csv(
    path: Path,
    column: str,
    counters: dict[str, Counter[int]],
    top_k: int,
) -> None:
    ranked = sorted(counters.items(), key=lambda item: sum(item[1].values()), reverse=True)[:top_k]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([column, "count", "positive_count", "negative_count", "positive_ratio"])
        for value, counter in ranked:
            total = counter.get(0, 0) + counter.get(1, 0)
            writer.writerow([value, total, counter.get(1, 0), counter.get(0, 0), f"{safe_divide(counter.get(1, 0), total):.6f}"])


def write_report_markdown(
    path: Path,
    *,
    row_count: int,
    target_counter: Counter[int],
    image_exists_counter: Counter[str],
    ordered_columns: list[str],
    missing_counter: Counter[str],
    numeric_values: dict[str, list[float]],
    leakage_groups: dict[str, list[str]],
) -> None:
    positive_count = target_counter.get(1, 0)
    negative_count = target_counter.get(0, 0)
    positive_ratio = safe_divide(positive_count, row_count)
    top_missing = sorted(
        ((column, missing_counter.get(column, 0)) for column in ordered_columns),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    lines = [
        "# ISIC2024 Tabular EDA Report",
        "",
        "## 데이터 개요",
        f"- 전체 행 수: `{row_count}`",
        f"- 양성 수: `{positive_count}`",
        f"- 음성 수: `{negative_count}`",
        f"- 양성 비율: `{positive_ratio:.6f}`",
        f"- 이미지 경로 확인 성공 수: `{image_exists_counter.get('1', 0)}`",
        f"- 이미지 경로 확인 실패 수: `{image_exists_counter.get('0', 0)}`",
        f"- 총 컬럼 수: `{len(ordered_columns)}`",
        "",
        "## 결측 상위 컬럼",
    ]
    for column, missing_count in top_missing:
        lines.append(f"- `{column}`: `{missing_count}`")

    lines.extend(
        [
            "",
            "## 수치형 후보 컬럼",
        ]
    )
    for column in sorted(NUMERIC_CANDIDATES):
        lines.append(f"- `{column}`: 유효값 `{len(numeric_values.get(column, []))}`개")

    lines.extend(
        [
            "",
            "## feature 그룹 분류 초안",
            "- 이 분류는 최종 확정이 아니라 EDA 해석을 위한 초안입니다.",
            f"- 초기 baseline 검토 가능: `{', '.join(leakage_groups['safe_for_initial_baseline_review'])}`",
            f"- 주의 검토 필요: `{', '.join(leakage_groups['caution_review_needed'])}`",
            f"- leakage 위험 높음: `{', '.join(leakage_groups['high_leakage_risk'])}`",
            "",
            "## 산출물",
            "- `dataset_overview.json`",
            "- `preview_rows.csv`",
            "- `missingness_summary.csv`",
            "- `numeric_summary.csv`",
            "- `categorical_summary.csv`",
            "- `target_rate_by_*.csv`",
            "- `feature_groups.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
