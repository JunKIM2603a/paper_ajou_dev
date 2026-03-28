from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from isic2024_benchmark.tabular_data import build_supplement_lookup, resolve_isic2024_dataset_root


@dataclass
class ImageSample:
    image_path: str
    label: int
    group_id: str
    isic_id: str
    source_split: str
    metadata: dict[str, Any]


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: list[ImageSample], image_size: int, augment: bool) -> None:
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.long)


# Backward-compatible alias for older imports.
CbisDdsmDataset = ImageClassificationDataset


def build_manifest(dataset_root: str | Path, cache_path: str | Path | None = None) -> list[dict[str, Any]]:
    paths = resolve_isic2024_dataset_root(dataset_root)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)

    supplement_lookup = build_supplement_lookup(paths)
    manifest: list[dict[str, Any]] = []
    with paths.ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            isic_id = row["isic_id"].strip()
            image_path = paths.image_dir / f"{isic_id}.jpg"
            if not image_path.exists():
                continue

            label = parse_binary_label(row["malignant"])
            supplement_row = supplement_lookup.get(isic_id, {})
            lesion_id = supplement_row.get("lesion_id", "").strip()
            group_id = lesion_id or isic_id

            manifest.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "group_id": group_id,
                    "isic_id": isic_id,
                    "source_split": "train_pool",
                    "metadata": {
                        "lesion_id": lesion_id,
                        "attribution": supplement_row.get("attribution", "").strip(),
                        "copyright_license": supplement_row.get("copyright_license", "").strip(),
                        "iddx_1": supplement_row.get("iddx_1", "").strip(),
                        "iddx_full": supplement_row.get("iddx_full", "").strip(),
                        "tbp_lv_dnn_lesion_confidence": supplement_row.get("tbp_lv_dnn_lesion_confidence", "").strip(),
                    },
                }
            )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    return manifest


def create_splits(
    manifest: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
    test_ratio: float = 0.2,
) -> dict[str, list[ImageSample]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in manifest:
        grouped[sample["group_id"]].append(sample)

    positive_groups: list[list[dict[str, Any]]] = []
    negative_groups: list[list[dict[str, Any]]] = []
    for group in grouped.values():
        group_label = max(item["label"] for item in group)
        if group_label == 1:
            positive_groups.append(group)
        else:
            negative_groups.append(group)

    random.Random(seed).shuffle(positive_groups)
    random.Random(seed + 1).shuffle(negative_groups)

    train_samples: list[ImageSample] = []
    val_samples: list[ImageSample] = []
    test_samples: list[ImageSample] = []

    for groups in [negative_groups, positive_groups]:
        test_count = max(1, int(len(groups) * test_ratio))
        remaining_groups = groups[test_count:]
        val_count = max(1, int(len(remaining_groups) * validation_ratio))

        test_part = groups[:test_count]
        val_part = remaining_groups[:val_count]
        train_part = remaining_groups[val_count:]

        for group in train_part:
            train_samples.extend(_to_image_samples(group, split_name="train"))
        for group in val_part:
            val_samples.extend(_to_image_samples(group, split_name="val"))
        for group in test_part:
            test_samples.extend(_to_image_samples(group, split_name="test"))

    return {"train": train_samples, "val": val_samples, "test": test_samples}


def parse_binary_label(value: str) -> int:
    normalized = value.strip()
    if normalized in {"1", "1.0"}:
        return 1
    return 0


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    return resolve_isic2024_dataset_root(dataset_root).dataset_root


def _to_image_samples(items: list[dict[str, Any]], split_name: str) -> list[ImageSample]:
    return [
        ImageSample(
            image_path=item["image_path"],
            label=item["label"],
            group_id=item["group_id"],
            isic_id=item["isic_id"],
            source_split=split_name,
            metadata=item["metadata"],
        )
        for item in items
    ]
