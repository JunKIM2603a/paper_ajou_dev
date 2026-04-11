from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2024_benchmark.runtime_env import ensure_expected_conda_env
from isic2024_benchmark.tabular_data import DEFAULT_DATASET_ROOT, load_tabular_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged ISIC2024 challenge tabular dataset for baseline modeling.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--feature-set-json", default="artifacts/eda/isic2024/feature_sets_recommended.json")
    parser.add_argument("--feature-set", choices=["strict", "relaxed", "oracle"], default="strict")
    parser.add_argument("--output", default="artifacts/tabular/isic2024_strict.csv")
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    args = parse_args()
    feature_set_payload = json.loads(Path(args.feature_set_json).read_text(encoding="utf-8"))
    target_column = feature_set_payload["target_column"]
    feature_columns = feature_set_payload["feature_sets"][args.feature_set]
    output_columns = feature_columns + [target_column]
    frame = load_tabular_dataframe(args.dataset_root, include_image_columns=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[output_columns].to_csv(output_path, index=False)

    print(f"Saved {args.feature_set} tabular dataset to {output_path}")


if __name__ == "__main__":
    main()
