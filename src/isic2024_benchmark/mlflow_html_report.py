from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from isic2024_benchmark.metrics import PRIMARY_PAUC_METRIC
from isic2024_benchmark.runtime_env import ensure_expected_conda_env, get_default_mlflow_tracking_uri
from isic2024_benchmark.tabular_terms import FEATURE_SET_DISPLAY_ORDER, feature_set_display_name, normalize_feature_set_name


PARENT_METRICS = [
    f"best_{PRIMARY_PAUC_METRIC}",
    "best_average_precision",
    "best_balanced_accuracy",
    "best_accuracy",
    "best_precision",
    "best_recall",
    "best_f1_score",
    "best_auc_roc",
]
CHILD_METRICS = [
    f"test_{PRIMARY_PAUC_METRIC}",
    "test_average_precision",
    "test_balanced_accuracy",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1_score",
    "test_auc_roc",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an HTML report from MLflow runs.")
    parser.add_argument("--tracking-uri", default=get_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Tabular-Benchmark")
    parser.add_argument("--output", default="artifacts/mlflow_report.html")
    parser.add_argument("--parent-sort-metric", default=f"best_{PRIMARY_PAUC_METRIC}")
    parser.add_argument("--child-sort-metric", default=f"test_{PRIMARY_PAUC_METRIC}")
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required to generate the HTML report.") from exc

    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {args.experiment_name}")

    parent_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.role = 'model_parent'",
        order_by=[f"metrics.{args.parent_sort_metric} DESC", "attributes.start_time DESC"],
    )
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.role = 'hyperparameter_trial'",
        order_by=[f"metrics.{args.child_sort_metric} DESC"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            experiment_id=experiment.experiment_id,
            parent_runs=parent_runs,
            child_runs=child_runs,
            parent_sort_metric=args.parent_sort_metric,
            child_sort_metric=args.child_sort_metric,
        ),
        encoding="utf-8",
    )
    print(f"Saved HTML report to {output_path}")


def build_html(
    *,
    experiment_name: str,
    tracking_uri: str,
    experiment_id: str,
    parent_runs,
    child_runs,
    parent_sort_metric: str,
    child_sort_metric: str,
) -> str:
    parent_records = select_best_parent_rows(parent_runs, parent_sort_metric)
    child_records = [row for _, row in child_runs.iterrows()]
    feature_set_records = select_best_child_rows_by_feature_set(child_records, child_sort_metric)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    for parent in parent_records:
        model_name = safe_value(parent.get("tags.model_name"))
        best_child_name = safe_value(parent.get("tags.best_child_run_name"))
        matching_children = [
            row for row in child_records if safe_value(row.get("tags.model_name")) == model_name
        ]
        sections.append(
            """
            <section class="panel">
              <div class="section-head">
                <div>
                  <h2>{model_name}</h2>
                  <p>Best child run: <strong>{best_child_name}</strong></p>
                </div>
                <div class="badge">Parent run: {parent_run_id}</div>
              </div>
              {parent_summary}
              {child_table}
            </section>
            """.format(
                model_name=escape(model_name),
                best_child_name=escape(best_child_name),
                parent_run_id=escape(safe_value(parent.get("run_id"))),
                parent_summary=build_parent_summary(parent, parent_sort_metric),
                child_table=build_child_table(matching_children, best_child_name, child_sort_metric),
            )
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLflow Report</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --panel: #fffaf3;
      --line: #d7c8b5;
      --text: #2f261d;
      --muted: #756553;
      --accent: #0f766e;
      --accent-soft: #d7f3ef;
      --best: #fff2bf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #fff6da 0, transparent 30%),
        radial-gradient(circle at top right, #dff6f2 0, transparent 28%),
        var(--bg);
    }}
    .wrap {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15,118,110,0.94), rgba(20,83,45,0.92));
      color: #f8fffd;
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(35, 30, 24, 0.16);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 6px 0;
      color: rgba(248,255,253,0.88);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .meta-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 12px 30px rgba(61, 46, 31, 0.08);
    }}
    .meta-card {{
      padding: 18px;
    }}
    .meta-card .label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .meta-card .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .panel {{
      margin-top: 22px;
      padding: 22px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .badge {{
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
      white-space: nowrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      overflow: hidden;
      border-radius: 14px;
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
      vertical-align: top;
    }}
    th {{
      background: #f1e4d3;
      color: #4a3d30;
      font-weight: 700;
    }}
    tr.best-row td {{
      background: var(--best);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}
    .summary-item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdf9;
    }}
    .summary-item .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-item .v {{
      margin-top: 8px;
      font-size: 22px;
      font-weight: 700;
    }}
    .empty {{
      margin-top: 12px;
      color: var(--muted);
    }}
    code {{
      font-family: Consolas, "SFMono-Regular", monospace;
      background: rgba(47, 38, 29, 0.06);
      padding: 1px 6px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>MLflow Experiment Report</h1>
      <p><strong>{experiment_name}</strong></p>
      <p>Experiment ID: <code>{experiment_id}</code></p>
      <p>Tracking URI: <code>{tracking_uri}</code></p>
      <p>Generated at: {generated_at}</p>
      <p>Parent sort metric: <code>{parent_sort_metric}</code> / Child sort metric: <code>{child_sort_metric}</code></p>
    </section>

    <section class="meta-grid">
      <article class="meta-card">
        <span class="label">Models</span>
        <div class="value">{model_count}</div>
      </article>
      <article class="meta-card">
        <span class="label">Parent Runs</span>
        <div class="value">{parent_count}</div>
      </article>
      <article class="meta-card">
        <span class="label">Trial Runs</span>
        <div class="value">{child_count}</div>
      </article>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Overall Leaderboard</h2>
          <p>Best parent runs sorted by <code>{parent_sort_metric}</code>.</p>
        </div>
      </div>
      {overall_leaderboard}
    </section>

    {feature_set_panels}

    {sections}
  </div>
</body>
</html>
""".format(
        experiment_name=escape(experiment_name),
        experiment_id=escape(experiment_id),
        tracking_uri=escape(tracking_uri),
        generated_at=escape(generated_at),
        parent_sort_metric=escape(parent_sort_metric),
        child_sort_metric=escape(child_sort_metric),
        model_count=len({safe_value(row.get("tags.model_name")) for row in parent_records}),
        parent_count=len(parent_records),
        child_count=len(child_records),
        overall_leaderboard=build_leaderboard(parent_records, parent_sort_metric),
        feature_set_panels=build_feature_set_panels(feature_set_records, child_sort_metric),
        sections="\n".join(sections) if sections else '<p class="empty">No runs found.</p>',
    )


def build_leaderboard(parent_records: list, parent_sort_metric: str) -> str:
    if not parent_records:
        return '<p class="empty">No parent runs found.</p>'

    rows = []
    for row in parent_records:
        rows.append(
            "<tr>"
            f"<td>{escape(safe_value(row.get('tags.model_name')))}</td>"
            f"<td>{escape(safe_value(row.get('params.best_feature_set')))}</td>"
            f"<td>{format_metric(row.get(f'metrics.{parent_sort_metric}'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_average_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_balanced_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_recall'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_f1_score'))}</td>"
            f"<td>{format_metric(row.get('metrics.best_auc_roc'))}</td>"
            f"<td>{escape(safe_value(row.get('tags.best_child_run_name')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Model</th><th>Feature Set</th><th>Primary</th><th>Avg Precision</th><th>Balanced Acc</th>"
        "<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Best Child</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_feature_set_panels(feature_set_records: dict[str, list], child_sort_metric: str) -> str:
    if not feature_set_records:
        return ""

    panels = []
    for feature_set, rows in feature_set_records.items():
        panels.append(
            """
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>{feature_set_label} Leaderboard</h2>
          <p>Best runs within <code>{feature_set}</code>, sorted by <code>{child_sort_metric}</code>.</p>
        </div>
        <div class="badge">{row_count} models</div>
      </div>
      {table}
    </section>
            """.format(
                feature_set_label=escape(feature_set_display_name(feature_set)),
                feature_set=escape(feature_set),
                child_sort_metric=escape(child_sort_metric),
                row_count=len(rows),
                table=build_feature_set_leaderboard(rows, child_sort_metric),
            )
        )
    return "\n".join(panels)


def build_feature_set_leaderboard(child_rows: list, child_sort_metric: str) -> str:
    if not child_rows:
        return '<p class="empty">No child runs found for this feature set.</p>'

    rows = []
    for index, row in enumerate(child_rows, start=1):
        row_class = ' class="best-row"' if index == 1 else ""
        params = {
            key.removeprefix("params.hp_"): safe_value(value)
            for key, value in row.items()
            if isinstance(key, str) and key.startswith("params.hp_")
        }
        rows.append(
            f"<tr{row_class}>"
            f"<td>{index}</td>"
            f"<td>{escape(safe_value(row.get('tags.model_name')))}</td>"
            f"<td>{escape(safe_value(row.get('tags.mlflow.runName')))}</td>"
            f"<td><code>{escape(json.dumps(params, ensure_ascii=False))}</code></td>"
            f"<td>{format_metric(row.get(f'metrics.{child_sort_metric}'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_average_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_balanced_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_recall'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_f1_score'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_auc_roc'))}</td>"
            f"<td>{format_metric(row.get('metrics.duration_seconds'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Rank</th><th>Model</th><th>Trial Run</th><th>Hyperparameters</th><th>Primary</th><th>Avg Precision</th>"
        "<th>Balanced Acc</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Duration (s)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_parent_summary(parent, parent_sort_metric: str) -> str:
    items = [("Primary", format_metric(parent.get(f"metrics.{parent_sort_metric}")))]
    items.extend(
        (label_from_metric(metric), format_metric(parent.get(f"metrics.{metric}")))
        for metric in PARENT_METRICS
        if metric != parent_sort_metric
    )
    items.append(("Feature Set", safe_value(parent.get("params.best_feature_set"))))
    return (
        '<div class="summary-grid">'
        + "".join(
            (
                '<div class="summary-item">'
                f'<div class="k">{escape(label)}</div>'
                f'<div class="v">{escape(value)}</div>'
                "</div>"
            )
            for label, value in items
        )
        + "</div>"
    )


def build_child_table(child_rows: list, best_child_name: str, child_sort_metric: str) -> str:
    if not child_rows:
        return '<p class="empty">No child runs found for this model.</p>'

    rows = []
    for row in child_rows:
        run_name = safe_value(row.get("tags.mlflow.runName"))
        row_class = ' class="best-row"' if run_name == best_child_name else ""
        params = {
            key.removeprefix("params.hp_"): safe_value(value)
            for key, value in row.items()
            if isinstance(key, str) and key.startswith("params.hp_")
        }
        rows.append(
            f"<tr{row_class}>"
            f"<td>{escape(run_name)}</td>"
            f"<td>{escape(safe_value(row.get('params.feature_set')))}</td>"
            f"<td><code>{escape(json.dumps(params, ensure_ascii=False))}</code></td>"
            f"<td>{format_metric(row.get(f'metrics.{child_sort_metric}'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_average_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_balanced_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_accuracy'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_precision'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_recall'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_f1_score'))}</td>"
            f"<td>{format_metric(row.get('metrics.test_auc_roc'))}</td>"
            f"<td>{format_metric(row.get('metrics.duration_seconds'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Trial Run</th><th>Feature Set</th><th>Hyperparameters</th><th>Primary</th><th>Avg Precision</th>"
        "<th>Balanced Acc</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Duration (s)</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def label_from_metric(metric_name: str) -> str:
    if PRIMARY_PAUC_METRIC in metric_name:
        return "pAUC @ TPR>=0.80"
    return metric_name.removeprefix("best_").replace("_", " ").title()


def select_best_parent_rows(parent_runs, parent_sort_metric: str) -> list:
    records = []
    seen_models: set[str] = set()
    metric_key = f"metrics.{parent_sort_metric}"
    for _, row in parent_runs.iterrows():
        model_name = safe_value(row.get("tags.model_name"))
        if not model_name or not safe_value(row.get(metric_key)):
            continue
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        records.append(row)
    return records


def select_best_child_rows_by_feature_set(child_records: list, child_sort_metric: str) -> dict[str, list]:
    metric_key = f"metrics.{child_sort_metric}"
    feature_sets = sorted(
        {
            normalize_feature_set_name(safe_value(row.get("params.feature_set")))
            for row in child_records
            if safe_value(row.get("params.feature_set"))
        },
        key=lambda name: (FEATURE_SET_DISPLAY_ORDER.get(name, len(FEATURE_SET_DISPLAY_ORDER)), name),
    )
    if not feature_sets:
        return {}

    grouped: dict[str, list] = {}
    for feature_set in feature_sets:
        selected = []
        seen_models: set[str] = set()
        for row in child_records:
            if normalize_feature_set_name(safe_value(row.get("params.feature_set"))) != feature_set:
                continue
            model_name = safe_value(row.get("tags.model_name"))
            if not model_name or not safe_value(row.get(metric_key)):
                continue
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            selected.append(row)
        grouped[feature_set] = selected
    return grouped


def safe_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def format_metric(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def escape(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
