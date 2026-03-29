from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from isic2024_benchmark.runtime_env import ensure_expected_conda_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a richer markdown EDA report with figures for ISIC2024.")
    parser.add_argument("--eda-dir", default="artifacts/eda/isic2024")
    parser.add_argument("--template", default="docs/isic2024_eda_report_template.md")
    parser.add_argument("--output", default="artifacts/eda/isic2024/eda_report.md")
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    args = parse_args()
    eda_dir = Path(args.eda_dir)
    template_path = Path(args.template)
    output_path = Path(args.output)
    figures_dir = eda_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overview = json.loads((eda_dir / "dataset_overview.json").read_text(encoding="utf-8"))
    feature_sets = json.loads((eda_dir / "feature_sets_recommended.json").read_text(encoding="utf-8"))
    missingness = pd.read_csv(eda_dir / "missingness_summary.csv")
    numeric = pd.read_csv(eda_dir / "numeric_summary.csv")
    iddx1 = pd.read_csv(eda_dir / "target_rate_by_iddx_1.csv")
    attribution = pd.read_csv(eda_dir / "target_rate_by_attribution.csv")
    categorical = pd.read_csv(eda_dir / "categorical_summary.csv")
    baseline = load_baseline_leaderboard()

    render_class_balance_figure(overview, figures_dir / "class_balance.png")
    render_missingness_figure(missingness, figures_dir / "missingness_top10.png")
    render_target_rate_figure(iddx1, "iddx_1", figures_dir / "target_rate_iddx1.png")
    render_target_rate_figure(attribution, "attribution", figures_dir / "target_rate_attribution.png")
    render_tbp_confidence_histogram(categorical, figures_dir / "tbp_confidence_hist.png")

    report = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{dataset_overview_table}}": build_dataset_overview_table(overview),
        "{{dataset_overview_interpretation}}": build_dataset_overview_interpretation(overview),
        "{{class_balance_interpretation}}": build_class_balance_interpretation(overview),
        "{{missingness_table}}": dataframe_to_markdown(
            missingness.sort_values("missing_ratio", ascending=False).head(10)[["column", "missing_count", "missing_ratio"]]
        ),
        "{{missingness_interpretation}}": build_missingness_interpretation(missingness),
        "{{iddx1_table}}": dataframe_to_markdown(iddx1[["iddx_1", "count", "positive_count", "positive_ratio"]]),
        "{{iddx1_interpretation}}": build_iddx1_interpretation(iddx1),
        "{{attribution_table}}": dataframe_to_markdown(
            attribution.sort_values("positive_ratio", ascending=False)[["attribution", "count", "positive_count", "positive_ratio"]]
        ),
        "{{attribution_interpretation}}": build_attribution_interpretation(attribution),
        "{{numeric_table}}": dataframe_to_markdown(numeric[["column", "group", "count", "mean", "median", "min", "max"]]),
        "{{numeric_interpretation}}": build_numeric_interpretation(numeric),
        "{{leakage_table}}": build_leakage_table(feature_sets),
        "{{leakage_interpretation}}": build_leakage_interpretation(feature_sets),
        "{{feature_set_table}}": build_feature_set_table(feature_sets),
        "{{feature_set_interpretation}}": build_feature_set_interpretation(feature_sets),
        "{{baseline_table}}": build_baseline_table(baseline),
        "{{discussion}}": build_discussion(overview, feature_sets, baseline),
        "{{conclusion}}": build_conclusion(overview, feature_sets),
    }

    for placeholder, value in replacements.items():
        report = report.replace(placeholder, value)

    output_path.write_text(report, encoding="utf-8")
    print(f"Saved rendered EDA report to {output_path}")


def render_class_balance_figure(overview: dict, output_path: Path) -> None:
    distribution = overview["target_distribution"]
    labels = ["negative", "positive"]
    values = [distribution["negative"], distribution["positive"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4C7A67", "#C65D4B"])
    ax.set_title("ISIC2024 Class Distribution")
    ax.set_ylabel("Count")
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_missingness_figure(missingness: pd.DataFrame, output_path: Path) -> None:
    top = missingness.sort_values("missing_ratio", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["column"], top["missing_ratio"], color="#D9863B")
    ax.set_title("Top 10 Missingness Ratios")
    ax.set_xlabel("Missing Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_target_rate_figure(frame: pd.DataFrame, label_column: str, output_path: Path) -> None:
    ordered = frame.sort_values("positive_ratio", ascending=False).head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(ordered)), ordered["positive_ratio"], color="#1E847F")
    ax.set_title(f"Positive Ratio by {label_column}")
    ax.set_ylabel("Positive Ratio")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(shorten_labels(ordered[label_column].tolist()), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_tbp_confidence_histogram(categorical: pd.DataFrame, output_path: Path) -> None:
    # 원본 EDA는 csv 요약 중심이라 figure는 merged csv를 다시 읽지 않고,
    # 현재 사용 가능한 가장 중요한 수치형 컬럼의 분포 설명용 텍스트 히스토그램 대신
    # 별도 데이터 로드 없는 구조를 유지하기 위해 placeholder 스타일 그래프를 피하고,
    # 실제 값 기반 figure가 필요하므로 export된 strict csv를 우선 활용한다.
    strict_csv = Path("artifacts/tabular/isic2024_strict.csv")
    if strict_csv.exists():
        frame = pd.read_csv(strict_csv)
        values = pd.to_numeric(frame["tbp_lv_dnn_lesion_confidence"], errors="coerce").dropna()
    else:
        values = pd.Series([], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(values) > 0:
        ax.hist(values, bins=40, color="#4F6D8A", edgecolor="white")
    ax.set_title("tbp_lv_dnn_lesion_confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_dataset_overview_table(overview: dict) -> str:
    rows = pd.DataFrame(
        [
            {"항목": "dataset_root", "값": overview["dataset_root"]},
            {"항목": "rows", "값": overview["rows"]},
            {"항목": "target_column", "값": overview["target_column"]},
            {"항목": "positive_count", "값": overview["target_distribution"]["positive"]},
            {"항목": "negative_count", "값": overview["target_distribution"]["negative"]},
            {"항목": "positive_ratio", "값": f"{overview['target_distribution']['positive_ratio']:.6f}"},
            {"항목": "column_count", "값": overview["column_count"]},
        ]
    )
    return dataframe_to_markdown(rows)


def build_dataset_overview_interpretation(overview: dict) -> str:
    positive_ratio = overview["target_distribution"]["positive_ratio"]
    return (
        f"본 데이터셋은 총 `{overview['rows']:,}`개의 표본으로 구성되어 있으며, 이 중 양성 비율은 `{positive_ratio:.6f}`에 불과하다. "
        "이는 일반적인 분류 데이터셋과 비교해도 매우 극단적인 불균형 조건에 해당한다. "
        "따라서 이후 baseline 결과를 해석할 때 단순 정확도보다는 양성 탐지 능력을 반영하는 지표를 우선적으로 살펴봐야 한다.\n\n"
        "또한 전체 컬럼 수는 많지 않지만, 컬럼의 성격은 균질하지 않다. 일부 변수는 메타데이터 수준의 보조 정보인 반면, "
        "일부 변수는 사실상 진단 결과와 매우 가까운 의미를 가진다. 이 때문에 이번 EDA의 핵심 목적은 단순 분포 요약이 아니라, "
        "어떤 컬럼을 메인 baseline feature로 허용할 수 있는지에 대한 판단 근거를 만드는 데 있다."
    )


def build_class_balance_interpretation(overview: dict) -> str:
    distribution = overview["target_distribution"]
    return (
        f"음성 표본은 `{distribution['negative']:,}`건인 반면 양성 표본은 `{distribution['positive']:,}`건에 불과하다. "
        "이 차이는 모델이 단순히 음성만 예측해도 매우 높은 정확도를 얻을 수 있음을 의미한다. "
        "즉, 이 문제에서 정확도는 모델이 실제로 병변을 잘 탐지하는지를 보여주는 대표 지표가 될 수 없다.\n\n"
        "따라서 목표 2에서 구성한 tabular baseline은 `best_average_precision`, `balanced_accuracy`, `recall`을 함께 확인하도록 설계했다. "
        "이는 모델이 얼마나 많은 양성을 실제로 포착하는지, 그리고 예측 점수의 순위가 얼마나 유의미한지를 동시에 보기 위함이다."
    )


def build_missingness_interpretation(missingness: pd.DataFrame) -> str:
    top = missingness.sort_values("missing_ratio", ascending=False).head(5)
    lines = [f"`{row.column}`의 결측률은 `{row.missing_ratio:.6f}`이다." for row in top.itertuples()]
    joined = " ".join(lines)
    return (
        f"{joined} 상위 결측 컬럼 대부분은 `iddx_*` 후반부와 `mel_*` 계열로 나타났다. "
        "이 패턴은 두 가지 해석을 가능하게 한다. 첫째, 이러한 변수는 데이터셋 전반에서 관측 가능한 일반 변수라기보다 "
        "특정 상황에서만 기록되는 후속 진단 정보일 가능성이 높다. 둘째, 실제 baseline feature로 사용할 경우 결측 처리 자체가 결과를 왜곡할 수 있다.\n\n"
        "또한 `patient_id`, `lesion_id`, `attribution`, `copyright_license`처럼 본질적 병변 특징보다 수집 단위나 출처에 가까운 컬럼은 "
        "단순 분포만 보면 유용해 보일 수 있어도, 공정한 일반화 비교 기준으로 쓰기에는 해석 리스크가 있다. "
        "이번 전환에서는 이런 컬럼을 `strict`에서 제외하고 `relaxed`에서만 별도 비교하도록 분리한다."
    )


def build_iddx1_interpretation(iddx1: pd.DataFrame) -> str:
    ordered = iddx1.sort_values("positive_ratio", ascending=False).reset_index(drop=True)
    highest = ordered.iloc[0]
    lowest = ordered.iloc[-1]
    return (
        f"`iddx_1={highest['iddx_1']}`의 양성 비율은 `{highest['positive_ratio']:.6f}`이고, "
        f"`iddx_1={lowest['iddx_1']}`은 `{lowest['positive_ratio']:.6f}`이다. "
        "이 차이는 단순 상관 수준을 넘어, `iddx_1`이 사실상 타깃과 매우 가까운 진단 정보를 포함하고 있음을 보여준다.\n\n"
        "즉, 이 변수는 메타데이터라기보다 이미 정리된 진단 판단 결과에 가깝다. "
        "따라서 `iddx_1`을 일반 baseline feature에 포함하면 모델이 입력 데이터를 학습하는 것이 아니라, "
        "이미 주어진 정답 힌트를 활용하는 구조가 된다. 이 때문에 본 프로젝트에서는 `iddx_1`을 `oracle` 세트에만 포함시키고, "
        "메인 비교에서는 제외하는 것이 타당하다."
    )


def build_attribution_interpretation(attribution: pd.DataFrame) -> str:
    highest = attribution.sort_values("positive_ratio", ascending=False).iloc[0]
    lowest = attribution.sort_values("positive_ratio", ascending=True).iloc[0]
    return (
        f"기관별 양성 비율은 최소 `{lowest['positive_ratio']:.6f}`에서 최대 `{highest['positive_ratio']:.6f}`까지 차이가 난다. "
        "이는 수집 기관에 따라 표본 구성과 난이도가 다를 수 있음을 시사한다. 다시 말해 `attribution`은 병변의 본질적 성질을 설명하는 변수라기보다, "
        "데이터가 어떤 환경에서 수집되었는지를 반영하는 변수일 가능성이 높다.\n\n"
        "`attribution`은 `iddx_*`처럼 직접적인 leakage 컬럼으로 보기는 어렵지만, 분포 차이를 통해 모델 성능에 간접적인 영향을 줄 수 있다. "
        "따라서 완전 제외보다는 `strict` 세트에 포함하되, 결과 해석 시 기관별 편향 가능성을 항상 함께 고려하는 것이 바람직하다."
    )


def build_numeric_interpretation(numeric: pd.DataFrame) -> str:
    tbp_pos = numeric[(numeric["column"] == "tbp_lv_dnn_lesion_confidence") & (numeric["group"] == "target_1")].iloc[0]
    tbp_neg = numeric[(numeric["column"] == "tbp_lv_dnn_lesion_confidence") & (numeric["group"] == "target_0")].iloc[0]
    mel = numeric[(numeric["column"] == "mel_thick_mm") & (numeric["group"] == "target_1")].iloc[0]
    return (
        f"`tbp_lv_dnn_lesion_confidence`의 평균은 음성 `{tbp_neg['mean']}`, 양성 `{tbp_pos['mean']}`로 차이가 나타난다. "
        "평균 차이만으로 모든 것이 설명되지는 않지만, 최소한 이 컬럼이 양성과 음성을 구분하는 데 일정 수준의 신호를 제공하고 있음을 보여준다. "
        "현재 `strict` 세트에서 가장 핵심적인 수치형 변수로 남는 이유도 여기에 있다.\n\n"
        f"반면 `mel_thick_mm`는 양성에서만 유효값 `{mel['count']}`개가 관측되었다. "
        "이런 패턴은 모델이 병변 특성을 학습한다기보다, 후속 진단 과정에서만 기록된 정보를 통해 양성을 맞히게 만들 수 있다. "
        "따라서 이 변수는 설명용으로는 의미가 있지만, 메인 baseline feature로 사용하기에는 위험하다. "
        "`mel_mitotic_index` 역시 유효값이 거의 없어 실제 학습 feature로서는 적절하지 않다."
    )


def build_leakage_table(feature_sets: dict) -> str:
    rows = []
    for label, columns in [
        ("excluded_columns", feature_sets["excluded_columns"]),
        ("high_leakage_risk_columns", feature_sets["high_leakage_risk_columns"]),
    ]:
        rows.append({"구분": label, "컬럼": ", ".join(columns)})
    return dataframe_to_markdown(pd.DataFrame(rows))


def build_leakage_interpretation(feature_sets: dict) -> str:
    return (
        "`iddx_*`, `iddx_full`은 계층형 진단 정보를 직접 담고 있어 leakage 위험이 높다. "
        "이 변수들을 포함했을 때 모델 성능이 과도하게 좋아진다면, 그것은 모델이 실제 예측 능력을 가진 것이 아니라 정답에 가까운 정보를 받아들였기 때문일 수 있다.\n\n"
        "`mel_thick_mm`, `mel_mitotic_index` 역시 관측 패턴이 일반형 입력 변수와 다르다. "
        "특히 유효값이 양성에만 집중되거나 거의 존재하지 않는 경우, 메인 baseline에서 사용하면 설명 가능성과 일반화 가능성을 동시에 해친다. "
        "따라서 본 프로젝트는 `strict`를 메인 비교 세트로, `oracle`을 leakage 상한선 확인용 세트로 명확히 분리한다."
    )


def build_feature_set_table(feature_sets: dict) -> str:
    rows = []
    for name, columns in feature_sets["feature_sets"].items():
        rows.append({"feature_set": name, "num_columns": len(columns), "columns": ", ".join(columns)})
    return dataframe_to_markdown(pd.DataFrame(rows))


def build_feature_set_interpretation(feature_sets: dict) -> str:
    return (
        "`strict`는 현실형 비교 기준으로, 실제 메인 결과표에 가장 적합하다. "
        "`relaxed`는 주의가 필요한 보조 정보까지 일부 포함하여, 어떤 컬럼이 성능을 얼마나 끌어올리는지 탐색하는 실험 세트다. "
        "`oracle`은 진단 계열 변수까지 포함하는 상한선 세트로, 모델 성능 자체보다는 leakage 영향의 크기를 보여주는 참고 기준으로 이해해야 한다."
    )

def build_baseline_table(baseline: pd.DataFrame) -> str:
    if baseline.empty:
        return "_baseline 결과 파일을 찾지 못했습니다._"
    visible = baseline[baseline["feature_set"].fillna("") != ""].copy()
    visible = visible[
        ["model_name", "feature_set", "best_average_precision", "balanced_accuracy", "recall", "auc_roc"]
    ]
    return dataframe_to_markdown(visible)


def build_discussion(overview: dict, feature_sets: dict, baseline: pd.DataFrame) -> str:
    if baseline.empty:
        return (
            "현재 baseline 리더보드가 없어 EDA와의 연결 논의를 자동 생성하지 못했다. "
            "baseline 실행 후 다시 렌더링하면 이 섹션이 채워진다."
        )

    visible = baseline[baseline["feature_set"].fillna("") != ""].copy()
    strict_best = visible[visible["feature_set"] == "strict"].sort_values("best_average_precision", ascending=False).head(1)
    relaxed_best = visible[visible["feature_set"] == "relaxed"].sort_values("best_average_precision", ascending=False).head(1)
    oracle_best = visible[visible["feature_set"] == "oracle"].sort_values("best_average_precision", ascending=False).head(1)

    parts = []
    if not strict_best.empty:
        row = strict_best.iloc[0]
        parts.append(
            f"`strict` 세트에서는 `{row['model_name']}`가 가장 높은 `best_average_precision={float(row['best_average_precision']):.6f}`를 기록했다. "
            "이는 극단적 불균형 환경에서도 제한된 메타데이터와 신뢰도 점수만으로 일정 수준의 순위화가 가능함을 보여준다."
        )
    if not relaxed_best.empty:
        row = relaxed_best.iloc[0]
        parts.append(
            f"`relaxed` 세트에서는 `{row['model_name']}`가 `best_average_precision={float(row['best_average_precision']):.6f}`를 기록했다. "
            "만약 이 값이 `strict` 대비 크게 높다면, `lesion_id`, `attribution`, `copyright_license` 같은 보조 메타데이터가 모델 성능에 강하게 개입하고 있음을 시사한다."
        )
    if not oracle_best.empty:
        row = oracle_best.iloc[0]
        parts.append(
            f"`oracle` 세트에서는 `{row['model_name']}`가 `best_average_precision={float(row['best_average_precision']):.6f}`를 기록했다. "
            "현재 결과에서 `oracle` 성능이 거의 완벽해지는 현상은, EDA에서 지적한 `iddx_*` 계열 leakage 위험이 실제 실험 결과로도 재확인되었음을 의미한다."
        )

    parts.append(
        "종합하면, EDA는 단순히 분포를 설명하는 단계에 그치지 않고, 어떤 feature set이 메인 baseline 비교 기준이 되어야 하는지 직접적인 실험 설계 근거를 제공했다. "
        "본 프로젝트에서는 `strict`를 메인 비교 기준으로 유지하고, `relaxed`와 `oracle`은 보조 해석 및 leakage 확인용으로 제한하는 것이 가장 일관된 선택이다."
    )
    return "\n\n".join(parts)


def build_conclusion(overview: dict, feature_sets: dict) -> str:
    return (
        "본 EDA는 `ISIC2024` tabular 데이터가 단순한 메타데이터 분류 문제가 아니라, "
        "극단적 클래스 불균형과 강한 leakage 후보가 동시에 존재하는 민감한 실험 환경임을 보여주었다. "
        "특히 `iddx_1`, `iddx_full`, `mel_thick_mm` 계열은 분포와 baseline 결과 모두에서 비정상적으로 강한 신호를 보였기 때문에, "
        "메인 비교 실험에 그대로 사용하는 것은 적절하지 않다.\n\n"
        "또한 동일 환자의 표본이 매우 많이 반복되므로, 이번 전환에서는 `patient_id` 기반 split을 통해 train/val/test 간 환자 중복을 차단하는 것이 핵심 전제다. "
        "따라서 현재 시점에서 가장 타당한 메인 baseline 기준은 `strict` feature set이다. "
        "`relaxed`와 `oracle`은 성능 향상 자체를 보고하기보다는, 어떤 컬럼이 결과를 과도하게 좋게 만드는지를 보여주는 보조 분석으로 활용하는 것이 적절하다. "
        "이 결론은 이후 image baseline 결과와 tabular baseline 결과를 공정하게 비교할 때도 중요한 기준점이 된다."
    )


def load_baseline_leaderboard() -> pd.DataFrame:
    path = Path("artifacts/tabular/mlflow_leaderboard.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    safe = frame.fillna("")
    columns = list(safe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in safe.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def shorten_labels(values: list[str], limit: int = 28) -> list[str]:
    shortened = []
    for value in values:
        if len(value) <= limit:
            shortened.append(value)
        else:
            shortened.append(value[: limit - 3] + "...")
    return shortened


if __name__ == "__main__":
    main()
