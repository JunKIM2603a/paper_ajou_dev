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
        "{{reading_guide}}": build_reading_guide(),
        "{{analysis_principles}}": build_analysis_principles(overview),
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
        "{{experiment_design_summary}}": build_experiment_design_summary(feature_sets),
        "{{column_policy_table}}": build_column_policy_table(feature_sets),
        "{{column_policy_interpretation}}": build_column_policy_interpretation(feature_sets),
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


def build_reading_guide() -> str:
    return (
        "> **핵심 방향**: 이 EDA의 목적은 보기 좋은 탐색 그 자체가 아니라, "
        "어떤 컬럼을 메인 baseline에 허용할지와 어떤 컬럼을 leakage 후보로 격리할지를 결정하는 데 있다.\n\n"
        "- `strict`는 메인 비교용 현실형 baseline이다.\n"
        "- `relaxed`는 보조 메타데이터가 성능에 미치는 영향을 점검하기 위한 보조 실험이다.\n"
        "- `oracle`은 leakage 상한선을 확인하기 위한 참고 세트다.\n"
        "- 따라서 이 보고서에서는 `성능이 높은가`보다 `어떤 정보 때문에 높아졌는가`를 더 중요하게 본다."
    )


def build_analysis_principles(overview: dict) -> str:
    return (
        "- **불균형 우선 해석**: 양성 비율이 매우 낮기 때문에 정확도보다 `average precision`, `AUC`, `balanced accuracy`, `recall`을 우선 본다.\n"
        "- **그룹 분리 우선**: 동일 환자 표본이 많으므로 `patient_id -> lesion_id -> isic_id` 정책으로 split을 해석한다.\n"
        "- **컬럼 허용 여부 우선**: 모든 컬럼을 동일하게 보지 않고, 식별자, 편향 가능 메타데이터, 진단 leakage, 일반 feature를 구분한다.\n"
        "- **공정한 비교 우선**: tabular 결과와 image 결과를 나중에 연결하기 위해, 메인 baseline은 설명 가능한 정보만 남긴 `strict`를 중심으로 설계한다.\n\n"
        f"> 현재 데이터는 총 `{overview['rows']:,}`건이며, 양성 비율은 `{overview['target_distribution']['positive_ratio']:.6f}`이다. "
        "이 조건에서는 EDA의 역할이 단순 요약보다 실험 설계 근거 정리에 더 가깝다."
    )


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
        "따라서 메인 비교용 `strict`에서는 제외하고, `relaxed`에서만 별도 비교하여 기관별 편향 가능성을 점검하는 것이 바람직하다."
    )


def build_numeric_interpretation(numeric: pd.DataFrame) -> str:
    tbp_pos = numeric[(numeric["column"] == "tbp_lv_dnn_lesion_confidence") & (numeric["group"] == "target_1")].iloc[0]
    tbp_neg = numeric[(numeric["column"] == "tbp_lv_dnn_lesion_confidence") & (numeric["group"] == "target_0")].iloc[0]
    mel = numeric[(numeric["column"] == "mel_thick_mm") & (numeric["group"] == "target_1")].iloc[0]
    return (
        f"`tbp_lv_dnn_lesion_confidence`의 평균은 음성 `{tbp_neg['mean']}`, 양성 `{tbp_pos['mean']}`로 차이가 나타난다. "
        "이 값은 악성 확률이라기보다 TBP 시스템이 해당 부위를 병변으로 얼마나 확신하는지를 나타내는 lesion confidence score에 가깝다. "
        "평균 차이만으로 모든 것이 설명되지는 않지만, 최소한 이 컬럼이 양성과 음성을 구분하는 데 일정 수준의 신호를 제공하고 있음을 보여준다. "
        "현재 `strict` 세트에서 유지되는 핵심 image-derived 수치형 변수로 남는 이유도 여기에 있다.\n\n"
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


def build_experiment_design_summary(feature_sets: dict) -> str:
    return (
        "> **실험 설계 관점 요약**\n"
        ">\n"
        "> 이 보고서는 컬럼을 모두 같은 수준의 입력으로 취급하지 않는다. "
        "식별자, 기관/출처 메타데이터, 진단 계열 정보, 일반 임상/TBP 수치형 feature를 분리해서 다루며, "
        "메인 baseline에는 설명 가능성과 공정성을 우선한다.\n"
        ">\n"
        "> 즉, EDA의 목표는 `무엇이 보이는가`를 나열하는 것이 아니라, "
        "`무엇을 학습에 허용할 것인가`를 결정하는 것이다."
    )


def _column_membership(feature_sets: dict, column: str) -> str:
    memberships = [name for name, columns in feature_sets["feature_sets"].items() if column in columns]
    if not memberships:
        return "excluded"
    return ", ".join(memberships)


def build_column_policy_table(feature_sets: dict) -> str:
    rows = [
        {
            "column_or_group": "`patient_id`, `isic_id`, `image_path`, `split_group_id`",
            "observation": "식별자 또는 분할 기준",
            "decision": "학습 입력으로 사용하지 않음",
            "placement": "excluded",
        },
        {
            "column_or_group": "`lesion_id`",
            "observation": "병변 식별 메타데이터",
            "decision": "메인 기준에서는 제외하고 편향 점검용으로만 사용",
            "placement": _column_membership(feature_sets, "lesion_id"),
        },
        {
            "column_or_group": "`attribution`, `copyright_license`",
            "observation": "기관/출처 메타데이터",
            "decision": "메타데이터 편향 가능성 점검용",
            "placement": _column_membership(feature_sets, "attribution"),
        },
        {
            "column_or_group": "`iddx_1`, `iddx_full`",
            "observation": "진단 결과와 거의 직접 연결",
            "decision": "leakage 상한선 확인용으로만 사용",
            "placement": _column_membership(feature_sets, "iddx_1"),
        },
        {
            "column_or_group": "`mel_thick_mm`, `mel_mitotic_index`",
            "observation": "후속 진단 계열 + 극단적 결측",
            "decision": "메인 baseline에는 부적절",
            "placement": _column_membership(feature_sets, "mel_thick_mm"),
        },
        {
            "column_or_group": "`tbp_lv_dnn_lesion_confidence`",
            "observation": "모든 샘플에 존재하는 lesion confidence score",
            "decision": "설명 가능한 핵심 수치형 신호로 유지",
            "placement": _column_membership(feature_sets, "tbp_lv_dnn_lesion_confidence"),
        },
        {
            "column_or_group": "기본 임상 + 대부분의 `tbp_lv_*`",
            "observation": "결측이 적고 일반 feature로 해석 가능",
            "decision": "메인 baseline 중심 입력",
            "placement": "strict, relaxed, oracle",
        },
    ]
    return dataframe_to_markdown(pd.DataFrame(rows))


def build_column_policy_interpretation(feature_sets: dict) -> str:
    strict_count = len(feature_sets["feature_sets"]["strict"])
    relaxed_count = len(feature_sets["feature_sets"]["relaxed"])
    oracle_count = len(feature_sets["feature_sets"]["oracle"])
    return (
        f"현재 추천안에서 `strict`는 `{strict_count}`개, `relaxed`는 `{relaxed_count}`개, `oracle`은 `{oracle_count}`개 컬럼으로 구성된다. "
        "이 차이는 단순히 컬럼 수를 늘리고 줄이는 문제가 아니라, 어떤 정보를 공정한 메인 비교에 허용할 것인지에 대한 정책 차이를 반영한다.\n\n"
        "특히 `strict`는 실제 성능을 보고하는 기준선이고, `relaxed`는 메타데이터 편향을 점검하는 보조 세트이며, "
        "`oracle`은 진단 leakage가 얼마나 큰지 확인하는 상한선이다. 따라서 `strict` 대비 `relaxed` 또는 `oracle`의 성능 상승은 "
        "모델 구조의 우수성이라기보다 입력 정보의 성격 변화로 먼저 해석해야 한다."
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
        "`oracle`은 진단 계열 변수까지 포함하는 상한선 세트로, 모델 성능 자체보다는 leakage 영향의 크기를 보여주는 참고 기준으로 이해해야 한다.\n\n"
        "> **해석 원칙**: `strict`보다 높은 성능이 관찰되더라도, 먼저 모델이 더 좋아졌다고 결론내리지 않고 "
        "어떤 추가 컬럼이 들어갔는지를 확인해야 한다."
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
