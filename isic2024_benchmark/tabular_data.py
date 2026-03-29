from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


CHALLENGE_METADATA_FILE = "train-metadata.csv"
CHALLENGE_IMAGE_DIR = Path("train-image") / "image"

LEGACY_GROUND_TRUTH_FILE = "ISIC_2024_Training_GroundTruth.csv"
LEGACY_SUPPLEMENT_FILE = "ISIC_2024_Training_Supplement.csv"
LEGACY_METADATA_FILE = Path("ISIC_2024_Training_Input") / "metadata.csv"
LEGACY_IMAGE_DIR = Path("ISIC_2024_Training_Input")

DEFAULT_DATASET_ROOT = Path("dataset") / "isic-2024-challenge"
DEFAULT_TARGET_COLUMN = "target"
IDENTIFIER_COLUMNS = {"isic_id", "patient_id", "lesion_id", "image_path", "image_exists", "split_group_id"}


@dataclass(frozen=True)
class Isic2024Paths:
    dataset_root: Path
    dataset_format: str
    image_dir: Path
    target_column: str
    metadata_csv: Path | None = None
    ground_truth_csv: Path | None = None
    supplement_csv: Path | None = None
    legacy_metadata_csv: Path | None = None


def resolve_isic2024_dataset_root(dataset_root: str | Path) -> Isic2024Paths:
    dataset_root = Path(dataset_root)
    candidate_roots = [dataset_root]

    if dataset_root.exists() and dataset_root.is_dir():
        candidate_roots.extend(
            child
            for child in dataset_root.iterdir()
            if child.is_dir() and child.name in {"isic-2024-challenge", "ISIC2024"}
        )
        candidate_roots.extend(
            child
            for child in dataset_root.iterdir()
            if child.is_dir() and "isic2024" in child.name.lower()
        )
        candidate_roots.extend(
            child
            for child in dataset_root.iterdir()
            if child.is_dir() and "isic-2024" in child.name.lower()
        )

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidate_roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        challenge_metadata = candidate / CHALLENGE_METADATA_FILE
        challenge_image_dir = candidate / CHALLENGE_IMAGE_DIR
        if challenge_metadata.exists() and challenge_image_dir.is_dir():
            return Isic2024Paths(
                dataset_root=candidate,
                dataset_format="challenge",
                metadata_csv=challenge_metadata,
                image_dir=challenge_image_dir,
                target_column=DEFAULT_TARGET_COLUMN,
            )

        legacy_ground_truth = candidate / LEGACY_GROUND_TRUTH_FILE
        legacy_supplement = candidate / LEGACY_SUPPLEMENT_FILE
        legacy_metadata = candidate / LEGACY_METADATA_FILE
        legacy_image_dir = candidate / LEGACY_IMAGE_DIR
        if legacy_ground_truth.exists() and legacy_supplement.exists() and legacy_metadata.exists() and legacy_image_dir.is_dir():
            return Isic2024Paths(
                dataset_root=candidate,
                dataset_format="legacy",
                ground_truth_csv=legacy_ground_truth,
                supplement_csv=legacy_supplement,
                legacy_metadata_csv=legacy_metadata,
                image_dir=legacy_image_dir,
                target_column="malignant",
            )

    searched = "\n".join(f"- {path.resolve()}" for path in ordered_candidates)
    raise FileNotFoundError(
        "ISIC2024 dataset not found.\n"
        "Expected either a challenge-format dataset root containing:\n"
        f"- {CHALLENGE_METADATA_FILE}\n"
        f"- {CHALLENGE_IMAGE_DIR}/\n"
        "or a legacy-format dataset root containing:\n"
        f"- {LEGACY_GROUND_TRUTH_FILE}\n"
        f"- {LEGACY_SUPPLEMENT_FILE}\n"
        f"- {LEGACY_METADATA_FILE}\n"
        f"Searched:\n{searched}"
    )


def iter_merged_tabular_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    if paths.dataset_format == "challenge":
        yield from _iter_challenge_rows(paths)
        return

    yield from _iter_legacy_rows(paths)


def load_tabular_dataframe(dataset_root: str | Path, *, include_image_columns: bool = True):
    import pandas as pd

    paths = resolve_isic2024_dataset_root(dataset_root)
    if paths.dataset_format == "challenge":
        if paths.metadata_csv is None:
            raise RuntimeError("Challenge metadata CSV is not configured.")
        frame = pd.read_csv(paths.metadata_csv, low_memory=False)
    else:
        if paths.ground_truth_csv is None or paths.supplement_csv is None or paths.legacy_metadata_csv is None:
            raise RuntimeError("Legacy dataset CSVs are not configured.")
        ground_truth = pd.read_csv(paths.ground_truth_csv, low_memory=False)
        metadata = pd.read_csv(paths.legacy_metadata_csv, low_memory=False)
        supplement = pd.read_csv(paths.supplement_csv, low_memory=False)
        frame = ground_truth.merge(metadata, on="isic_id", how="inner", validate="1:1")
        frame = frame.merge(supplement, on="isic_id", how="left", validate="1:1")
        if "malignant" in frame.columns and DEFAULT_TARGET_COLUMN not in frame.columns:
            frame = frame.rename(columns={"malignant": DEFAULT_TARGET_COLUMN})

    frame.columns = [str(column).strip() for column in frame.columns]
    frame["isic_id"] = _normalize_series(frame["isic_id"])

    if include_image_columns:
        existing_ids = {path.stem for path in paths.image_dir.glob("*.jpg")}
        frame["image_path"] = frame["isic_id"].map(lambda isic_id: str(paths.image_dir / f"{isic_id}.jpg"))
        frame["image_exists"] = frame["isic_id"].isin(existing_ids).astype(int)

    frame[DEFAULT_TARGET_COLUMN] = pd.to_numeric(frame[DEFAULT_TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    frame["split_group_id"] = build_split_group_ids(frame)
    return frame


def build_split_group_ids(frame):
    patient_ids = _normalize_series(frame["patient_id"]) if "patient_id" in frame.columns else None
    lesion_ids = _normalize_series(frame["lesion_id"]) if "lesion_id" in frame.columns else None
    isic_ids = _normalize_series(frame["isic_id"])

    if patient_ids is not None:
        group_ids = patient_ids.where(patient_ids != "", other=None)
    else:
        group_ids = None
    if lesion_ids is not None:
        if group_ids is None:
            group_ids = lesion_ids.where(lesion_ids != "", other=None)
        else:
            group_ids = group_ids.where(group_ids.notna(), lesion_ids.where(lesion_ids != "", other=None))
    if group_ids is None:
        return isic_ids
    return group_ids.where(group_ids.notna(), isic_ids)


def build_group_id_from_row(row: dict[str, str]) -> str:
    patient_id = normalize_cell(row.get("patient_id", ""))
    if patient_id:
        return patient_id
    lesion_id = normalize_cell(row.get("lesion_id", ""))
    if lesion_id:
        return lesion_id
    return normalize_cell(row.get("isic_id", ""))


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _iter_challenge_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    if paths.metadata_csv is None:
        raise RuntimeError("Challenge metadata CSV is not configured.")

    with paths.metadata_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            row = {str(key).strip(): normalize_cell(value) for key, value in raw_row.items()}
            isic_id = row["isic_id"]
            image_path = paths.image_dir / f"{isic_id}.jpg"
            row["image_path"] = str(image_path)
            row["image_exists"] = "1" if image_path.exists() else "0"
            if "malignant" in row and DEFAULT_TARGET_COLUMN not in row:
                row[DEFAULT_TARGET_COLUMN] = row.pop("malignant")
            yield row


def _iter_legacy_rows(paths: Isic2024Paths) -> Iterator[dict[str, str]]:
    if paths.ground_truth_csv is None or paths.supplement_csv is None or paths.legacy_metadata_csv is None:
        raise RuntimeError("Legacy dataset CSVs are not configured.")

    metadata_lookup = _build_lookup(paths.legacy_metadata_csv)
    supplement_lookup = _build_lookup(paths.supplement_csv)
    with paths.ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            ground_truth_row = {str(key).strip(): normalize_cell(value) for key, value in raw_row.items()}
            isic_id = ground_truth_row["isic_id"]
            merged = {
                "isic_id": isic_id,
                DEFAULT_TARGET_COLUMN: ground_truth_row.get("malignant", ""),
            }
            metadata_row = metadata_lookup.get(isic_id, {})
            supplement_row = supplement_lookup.get(isic_id, {})
            for source_row in (metadata_row, supplement_row):
                for key, value in source_row.items():
                    if key == "isic_id":
                        continue
                    merged[key] = value
            image_path = paths.image_dir / f"{isic_id}.jpg"
            merged["image_path"] = str(image_path)
            merged["image_exists"] = "1" if image_path.exists() else "0"
            yield merged


def _build_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for raw_row in csv.DictReader(file):
            row = {str(key).strip(): normalize_cell(value) for key, value in raw_row.items()}
            lookup[row["isic_id"]] = row
    return lookup


def _normalize_series(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": "", "<na>": ""})
    )
