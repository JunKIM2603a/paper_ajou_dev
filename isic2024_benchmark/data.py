from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from isic2024_benchmark.split_utils import split_group_ids
from isic2024_benchmark.tabular_data import (
    build_group_id_from_row,
    iter_merged_tabular_rows,
    normalize_cell,
    resolve_isic2024_dataset_root,
)


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


CbisDdsmDataset = ImageClassificationDataset


def build_manifest(dataset_root: str | Path, cache_path: str | Path | None = None) -> list[dict[str, Any]]:
    paths = resolve_isic2024_dataset_root(dataset_root)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)

    manifest: list[dict[str, Any]] = []
    for row in iter_merged_tabular_rows(paths):
        isic_id = normalize_cell(row.get("isic_id", ""))
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue

        label = parse_binary_label(row.get(paths.target_column, row.get("target", "0")))
        group_id = build_group_id_from_row(row)
        manifest.append(
            {
                "image_path": str(image_path),
                "label": label,
                "group_id": group_id,
                "isic_id": isic_id,
                "source_split": "train_pool",
                "metadata": {
                    "patient_id": normalize_cell(row.get("patient_id", "")),
                    "lesion_id": normalize_cell(row.get("lesion_id", "")),
                    "attribution": normalize_cell(row.get("attribution", "")),
                    "copyright_license": normalize_cell(row.get("copyright_license", "")),
                    "iddx_1": normalize_cell(row.get("iddx_1", "")),
                    "iddx_full": normalize_cell(row.get("iddx_full", "")),
                    "tbp_lv_dnn_lesion_confidence": normalize_cell(row.get("tbp_lv_dnn_lesion_confidence", "")),
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
    group_labels: dict[str, int] = {}
    for sample in manifest:
        group_id = str(sample["group_id"])
        group_labels[group_id] = max(group_labels.get(group_id, 0), int(sample["label"]))

    split_ids = split_group_ids(
        group_labels,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_items = {"train": [], "val": [], "test": []}
    for sample in manifest:
        group_id = str(sample["group_id"])
        if group_id in split_ids["train"]:
            split_name = "train"
        elif group_id in split_ids["val"]:
            split_name = "val"
        else:
            split_name = "test"
        split_items[split_name].append(sample)

    return {split_name: _to_image_samples(items, split_name=split_name) for split_name, items in split_items.items()}


def parse_binary_label(value: str) -> int:
    normalized = normalize_cell(value)
    if normalized in {"1", "1.0", "true", "True"}:
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
