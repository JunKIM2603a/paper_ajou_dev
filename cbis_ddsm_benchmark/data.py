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


BENIGN_LABELS = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}
MALIGNANT_LABELS = {"MALIGNANT"}


@dataclass
class Sample:
    image_path: str
    label: int
    patient_id: str
    pathology: str
    abnormality_type: str
    source_split: str
    metadata: dict[str, Any]


class CbisDdsmDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, augment: bool) -> None:
        # 학습 시에는 약한 증강을 적용하고, 검증/테스트 시에는 입력을 고정한다.
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.long)


# CBIS-DDSM의 CSV 설명 파일과 실제 JPEG 크롭 이미지를 연결해 학습용 매니페스트를 만든다.
def build_manifest(dataset_root: str | Path, cache_path: str | Path | None = None) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)

    csv_root = dataset_root / "csv"
    jpeg_root = dataset_root / "jpeg"

    cropped_lookup = _build_cropped_lookup(csv_root / "dicom_info.csv", jpeg_root)
    manifest: list[dict[str, Any]] = []
    csv_files = [
        ("train", csv_root / "mass_case_description_train_set.csv"),
        ("test", csv_root / "mass_case_description_test_set.csv"),
        ("train", csv_root / "calc_case_description_train_set.csv"),
        ("test", csv_root / "calc_case_description_test_set.csv"),
    ]

    for split_name, path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pathology = row["pathology"].strip()
                label = pathology_to_label(pathology)
                if label is None:
                    continue

                # ROI mask가 아닌 병변 크롭 이미지를 기준으로 분류 입력을 만든다.
                cropped_file = row["cropped image file path"].strip()
                cropped_series_uid = _extract_series_uid(cropped_file)
                image_path = cropped_lookup.get(cropped_series_uid)
                if image_path is None or not image_path.exists():
                    continue

                abnormality_type = row.get("abnormality type", "").strip()
                manifest.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "patient_id": row["patient_id"].strip(),
                        "pathology": pathology,
                        "abnormality_type": abnormality_type,
                        "source_split": split_name,
                        "metadata": {
                            "assessment": row.get("assessment", "").strip(),
                            "image_view": row.get("image view", "").strip(),
                            "laterality": row.get("left or right breast", "").strip(),
                            "abnormality_id": row.get("abnormality id", "").strip(),
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
) -> dict[str, list[Sample]]:
    # 같은 환자의 이미지가 train/val에 동시에 섞이지 않도록 patient 단위로 묶는다.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in manifest:
        grouped[sample["patient_id"]].append(sample)

    train_groups: list[list[dict[str, Any]]] = []
    test_samples: list[Sample] = []

    for patient_samples in grouped.values():
        sample_split = patient_samples[0]["source_split"]
        if sample_split == "test":
            test_samples.extend(_to_samples(patient_samples))
        else:
            train_groups.append(patient_samples)

    # 클래스 비율이 너무 치우치지 않도록 benign / malignant 그룹을 따로 섞는다.
    benign_groups: list[list[dict[str, Any]]] = []
    malignant_groups: list[list[dict[str, Any]]] = []
    for group in train_groups:
        group_label = max(sample["label"] for sample in group)
        if group_label == 1:
            malignant_groups.append(group)
        else:
            benign_groups.append(group)

    random.Random(seed).shuffle(benign_groups)
    random.Random(seed + 1).shuffle(malignant_groups)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for class_groups in [benign_groups, malignant_groups]:
        val_count = max(1, int(len(class_groups) * validation_ratio))
        val_groups = class_groups[:val_count]
        train_part = class_groups[val_count:]
        for group in train_part:
            train_samples.extend(_to_samples(group))
        for group in val_groups:
            val_samples.extend(_to_samples(group))

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def pathology_to_label(pathology: str) -> int | None:
    if pathology in BENIGN_LABELS:
        return 0
    if pathology in MALIGNANT_LABELS:
        return 1
    return None


# dicom_info.csv에서 cropped images만 골라 series UID -> JPEG 경로 매핑을 만든다.
def _build_cropped_lookup(dicom_info_path: Path, jpeg_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    with dicom_info_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = row["image_path"].strip()
            if not image_path:
                continue
            series_uid = Path(image_path).parent.name
            if row.get("SeriesDescription", "").strip().lower() != "cropped images":
                continue
            relative = Path(*Path(image_path).parts[2:])
            lookup[series_uid] = jpeg_root / relative
    return lookup


def _extract_series_uid(cropped_path: str) -> str:
    parts = Path(cropped_path).parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected cropped image path format: {cropped_path}")
    return parts[-2]


def _to_samples(items: list[dict[str, Any]]) -> list[Sample]:
    return [
        Sample(
            image_path=item["image_path"],
            label=item["label"],
            patient_id=item["patient_id"],
            pathology=item["pathology"],
            abnormality_type=item["abnormality_type"],
            source_split=item["source_split"],
            metadata=item["metadata"],
        )
        for item in items
    ]








