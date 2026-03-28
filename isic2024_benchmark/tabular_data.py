from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


GROUND_TRUTH_FILE = "ISIC_2024_Training_GroundTruth.csv"
SUPPLEMENT_FILE = "ISIC_2024_Training_Supplement.csv"
IMAGE_DIR = "ISIC_2024_Training_Input"


@dataclass(frozen=True)
class Isic2024Paths:
    dataset_root: Path
    ground_truth_csv: Path
    supplement_csv: Path
    image_dir: Path


def resolve_isic2024_dataset_root(dataset_root: str | Path) -> Isic2024Paths:
    dataset_root = Path(dataset_root)
    candidate_roots = [dataset_root]

    if dataset_root.exists() and dataset_root.is_dir():
        candidate_roots.extend(
            child for child in dataset_root.iterdir() if child.is_dir() and "isic2024" in child.name.lower()
        )
        candidate_roots.extend(
            child for child in dataset_root.iterdir() if child.is_dir() and child.name == "ISIC2024"
        )

    for candidate in candidate_roots:
        ground_truth_csv = candidate / GROUND_TRUTH_FILE
        supplement_csv = candidate / SUPPLEMENT_FILE
        image_dir = candidate / IMAGE_DIR
        if ground_truth_csv.exists() and supplement_csv.exists() and image_dir.is_dir():
            return Isic2024Paths(
                dataset_root=candidate,
                ground_truth_csv=ground_truth_csv,
                supplement_csv=supplement_csv,
                image_dir=image_dir,
            )

    searched = "\n".join(f"- {path.resolve()}" for path in candidate_roots)
    raise FileNotFoundError(
        "ISIC2024 dataset not found.\n"
        "Expected a dataset root containing these paths:\n"
        f"- {GROUND_TRUTH_FILE}\n"
        f"- {SUPPLEMENT_FILE}\n"
        f"- {IMAGE_DIR}/\n"
        f"Searched:\n{searched}"
    )


def read_ground_truth_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    with paths.ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def read_supplement_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    with paths.supplement_csv.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def build_supplement_lookup(paths: Isic2024Paths) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_supplement_rows(paths):
        isic_id = row["isic_id"].strip()
        lookup[isic_id] = {key: value.strip() for key, value in row.items()}
    return lookup


def iter_merged_tabular_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    supplement_lookup = build_supplement_lookup(paths)
    for ground_truth_row in read_ground_truth_rows(paths):
        isic_id = ground_truth_row["isic_id"].strip()
        supplement_row = supplement_lookup.get(isic_id, {})
        image_path = paths.image_dir / f"{isic_id}.jpg"
        merged = {
            "isic_id": isic_id,
            "image_path": str(image_path),
            "image_exists": "1" if image_path.exists() else "0",
            "malignant": ground_truth_row["malignant"].strip(),
        }
        for key, value in supplement_row.items():
            if key == "isic_id":
                continue
            merged[key] = value
        yield merged
