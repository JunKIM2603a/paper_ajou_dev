from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from isic2024_multimodal.data.splits import split_group_ids
from isic2024_multimodal.data.tabular_dataset import (
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
    def __init__(
        self,
        samples: list[ImageSample],
        image_size: int,
        augment: bool,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        normalize_std = normalize_std or [0.229, 0.224, 0.225]
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=normalize_mean, std=normalize_std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=normalize_mean, std=normalize_std),
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


def create_splits_from_locked_csvs(
    manifest: list[dict[str, Any]],
    *,
    holdout_split_csv: str | Path,
    cv_split_csv: str | Path,
    cv_fold: int,
) -> dict[str, list[ImageSample]]:
    holdout_path = Path(holdout_split_csv)
    cv_path = Path(cv_split_csv)
    if not holdout_path.exists() or not cv_path.exists():
        missing = [str(path) for path in [holdout_path, cv_path] if not path.exists()]
        raise FileNotFoundError(
            "Locked split CSV files are required for paper-valid image baselines. "
            f"Missing: {missing}"
        )

    holdout_rows = read_csv_rows(holdout_path)
    cv_rows = read_csv_rows(cv_path)
    train_validation_ids = {
        str(row["isic_id"])
        for row in holdout_rows
        if str(row.get("split", "")) == "train_validation_data"
    }
    test_ids = {str(row["isic_id"]) for row in holdout_rows if str(row.get("split", "")) == "test_data"}
    val_ids = {
        str(row["isic_id"])
        for row in cv_rows
        if int(row.get("cv_validation_fold", -1)) == int(cv_fold)
    }
    train_ids = train_validation_ids - val_ids
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "Locked split CSVs produced an empty image split: "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    patient_lookup = {str(row["isic_id"]): str(row.get("patient_id", "")) for row in holdout_rows}
    overlap_checks = locked_split_patient_overlap(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        patient_lookup=patient_lookup,
    )
    failed_checks = {key: value for key, value in overlap_checks.items() if value != 0}
    if failed_checks:
        raise RuntimeError(f"Locked image split patient overlap audit failed: {failed_checks}")

    split_items = {"train": [], "val": [], "test": []}
    missing_manifest_ids = {"train": 0, "val": 0, "test": 0}
    manifest_ids = {str(sample["isic_id"]) for sample in manifest}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        missing_manifest_ids[split_name] = len(ids - manifest_ids)
    for sample in manifest:
        isic_id = str(sample["isic_id"])
        if isic_id in train_ids:
            split_items["train"].append(sample)
        elif isic_id in val_ids:
            split_items["val"].append(sample)
        elif isic_id in test_ids:
            split_items["test"].append(sample)

    if any(not split_items[name] for name in split_items):
        raise RuntimeError(
            "Locked split has no image samples after manifest filtering: "
            f"train={len(split_items['train'])}, val={len(split_items['val'])}, test={len(split_items['test'])}, "
            f"missing_manifest_ids={missing_manifest_ids}"
        )
    return {split_name: _to_image_samples(items, split_name=split_name) for split_name, items in split_items.items()}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def locked_split_patient_overlap(
    *,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    patient_lookup: dict[str, str],
) -> dict[str, int]:
    train_patients = {patient_lookup.get(isic_id, "") for isic_id in train_ids}
    val_patients = {patient_lookup.get(isic_id, "") for isic_id in val_ids}
    test_patients = {patient_lookup.get(isic_id, "") for isic_id in test_ids}
    train_patients.discard("")
    val_patients.discard("")
    test_patients.discard("")
    return {
        "train_val_patient_overlap": len(train_patients & val_patients),
        "train_test_patient_overlap": len(train_patients & test_patients),
        "val_test_patient_overlap": len(val_patients & test_patients),
    }


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
