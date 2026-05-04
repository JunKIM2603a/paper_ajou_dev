from __future__ import annotations

import argparse
import json
from pathlib import Path

from isic2024_multimodal.models.image.checkpoint_downloads import CHECKPOINT_DOWNLOADS, download_checkpoint
from isic2024_multimodal.utils.config_utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download registered image-baseline checkpoints.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to download, e.g. chexzero eyepacs medclip retfound. Defaults to all registered downloads.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional image baseline config JSON. When set, downloads the checkpoint required by that config.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root used to resolve checkpoint paths.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the target file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.config:
        config = load_json(args.config)
        from isic2024_multimodal.models.image.checkpoint_downloads import download_for_model_config

        reports = [download_for_model_config(config["model"], repo_root=repo_root, force=args.force)]
    else:
        models = args.models or sorted(CHECKPOINT_DOWNLOADS)
        reports = [
            download_checkpoint(model_key, repo_root=repo_root, force=args.force)
            for model_key in models
        ]

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
