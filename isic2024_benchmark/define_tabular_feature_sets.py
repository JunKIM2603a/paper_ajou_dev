from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2024_benchmark.runtime_env import ensure_expected_conda_env
from isic2024_benchmark.tabular_feature_sets import recommend_feature_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Define tabular feature sets from EDA artifacts.")
    parser.add_argument("--eda-dir", default="artifacts/eda/isic2024")
    parser.add_argument("--output", default="artifacts/eda/isic2024/feature_sets_recommended.json")
    return parser.parse_args()


def main() -> None:
    ensure_expected_conda_env()
    args = parse_args()
    recommendations = recommend_feature_sets(args.eda_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved tabular feature set recommendations to {output_path}")


if __name__ == "__main__":
    main()
