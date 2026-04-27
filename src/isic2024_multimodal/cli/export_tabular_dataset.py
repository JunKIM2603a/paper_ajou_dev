from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2024_multimodal.features.final_tabular_inputs import build_final_feature_frames, is_final_inputs_feature_payload
from isic2024_multimodal.utils.runtime_env import ensure_expected_conda_env
from isic2024_multimodal.data.tabular_dataset import DEFAULT_DATASET_ROOT, load_tabular_dataframe
from isic2024_multimodal.features.tabular_terms import ORACLE, RELAXED, STRICT_BASE, STRICT_FE, STRICT_MAIN_INPUT, normalize_feature_set_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged ISIC2024 challenge tabular dataset for baseline modeling.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--feature-set-json", default="experiments/evidence/eda/isic_2024/final_inputs/feature_sets_recommended.json")
    parser.add_argument("--eda-dir", default="experiments/evidence/eda/isic_2024")
    parser.add_argument(
        "--feature-set",
        choices=[STRICT_BASE, STRICT_FE, STRICT_MAIN_INPUT, RELAXED, ORACLE, "strict"],
        default=STRICT_MAIN_INPUT,
    )
    parser.add_argument("--output", default="experiments/tables/isic2024_strict_main_input.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_expected_conda_env()
    feature_set_payload = json.loads(Path(args.feature_set_json).read_text(encoding="utf-8"))
    feature_set_payload["feature_sets"] = {
        normalize_feature_set_name(name): columns for name, columns in feature_set_payload.get("feature_sets", {}).items()
    }
    normalized_feature_set = normalize_feature_set_name(args.feature_set)
    target_column = feature_set_payload["target_column"]
    feature_columns = feature_set_payload["feature_sets"][normalized_feature_set]
    output_columns = feature_columns + [target_column]
    frame = load_tabular_dataframe(args.dataset_root, include_image_columns=False)

    if is_final_inputs_feature_payload(feature_set_payload):
        export_frame = build_final_feature_frames(frame, args.eda_dir, [normalized_feature_set])[normalized_feature_set].copy()
        export_frame[target_column] = frame[target_column].values
    else:
        export_frame = frame[output_columns].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame[output_columns].to_csv(output_path, index=False)

    print(f"Saved {normalized_feature_set} tabular dataset to {output_path}")


if __name__ == "__main__":
    main()
