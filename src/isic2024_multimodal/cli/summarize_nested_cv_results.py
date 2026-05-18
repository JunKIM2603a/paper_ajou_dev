from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2024_multimodal.experiments.families import EXPERIMENT_FAMILIES, resolve_family_paths
from isic2024_multimodal.experiments.nested_cv_summary import (
    DEFAULT_VALIDATION_METRIC_ORDER,
    collect_nested_cv_summary_records,
    write_nested_cv_summary_outputs,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Git-friendly nested-CV result tables from local baseline summary.json files."
    )
    parser.add_argument("--family", default="tabular_baselines", choices=sorted(EXPERIMENT_FAMILIES))
    parser.add_argument(
        "--run-group-id",
        default=None,
        help="Run group id. Required unless --output-root is provided; otherwise inferred from output root name.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory that contains nested run outputs. Defaults to experiments/outputs/<family>/<run_group_id>.",
    )
    parser.add_argument(
        "--table-root",
        default=None,
        help="Directory for small summary artifacts. Defaults to experiments/tables/<family>/<run_group_id>/nested_cv.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use smoke family path defaults.")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument(
        "--validation-metrics",
        nargs="*",
        default=DEFAULT_VALIDATION_METRIC_ORDER,
        help="Validation metric priority used for outer-fold candidate selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root, table_root, run_group_id = resolve_summary_paths(args)
    records = collect_nested_cv_summary_records(
        output_root=output_root,
        family=args.family,
        run_group_id=run_group_id,
        validation_metric_order=args.validation_metrics,
    )
    if not records:
        raise SystemExit(f"No nested-CV summary records found under: {output_root}")

    manifest = write_nested_cv_summary_outputs(
        records=records,
        table_root=table_root,
        family=args.family,
        run_group_id=run_group_id,
        expected_outer_folds=args.expected_outer_folds,
        validation_metric_order=args.validation_metrics,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def resolve_summary_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    run_group_id = args.run_group_id
    output_root = resolve_repo_path(args.output_root) if args.output_root else None
    if run_group_id is None and output_root is not None:
        run_group_id = output_root.name
    if run_group_id is None:
        raise SystemExit("Provide --run-group-id, or provide --output-root so the run group id can be inferred.")

    family_paths = resolve_family_paths(
        family=args.family,
        run_group_id=run_group_id,
        repo_root=REPO_ROOT,
        smoke=args.smoke,
    )
    output_root = output_root or family_paths.output_root
    table_root = resolve_repo_path(args.table_root) if args.table_root else family_paths.table_root / "nested_cv"
    return output_root, table_root, run_group_id


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


if __name__ == "__main__":
    main()
