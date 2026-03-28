from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isic2024_benchmark.tabular_data import iter_merged_tabular_rows, resolve_isic2024_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged ISIC2024 tabular dataset for baseline modeling.")
    parser.add_argument("--dataset-root", default="dataset/ISIC2024")
    parser.add_argument("--feature-set-json", default="artifacts/eda/isic2024/feature_sets_recommended.json")
    parser.add_argument("--feature-set", choices=["strict", "relaxed", "oracle"], default="strict")
    parser.add_argument("--output", default="artifacts/tabular/isic2024_strict.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_isic2024_dataset_root(args.dataset_root)
    feature_set_payload = json.loads(Path(args.feature_set_json).read_text(encoding="utf-8"))
    target_column = feature_set_payload["target_column"]
    feature_columns = feature_set_payload["feature_sets"][args.feature_set]
    output_columns = feature_columns + [target_column]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        for row in iter_merged_tabular_rows(paths):
            writer.writerow({column: row.get(column, "") for column in output_columns})

    print(f"Saved {args.feature_set} tabular dataset to {output_path}")


if __name__ == "__main__":
    main()
