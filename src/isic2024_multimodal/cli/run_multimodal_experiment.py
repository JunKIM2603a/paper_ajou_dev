from __future__ import annotations

import argparse

from isic2024_multimodal.data.tabular_dataset import DEFAULT_DATASET_ROOT
from isic2024_multimodal.utils.runtime_env import get_default_mlflow_tracking_uri

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ISIC2024 image-tabular multimodal experiment.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--config", default="experiments/configs/multimodal/default.json")
    parser.add_argument("--output-root", default="experiments/outputs/multimodal")
    parser.add_argument("--tracking-uri", default=get_default_mlflow_tracking_uri())
    parser.add_argument("--experiment-name", default="ISIC2024-Multimodal")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-spec", default=None)
    parser.add_argument("--model-family", default="multimodal_baselines")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError(
        "Multimodal training is not implemented yet. Use image and tabular baseline CLIs first."
    )


if __name__ == "__main__":
    main()
