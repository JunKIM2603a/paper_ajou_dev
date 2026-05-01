from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isic2024_multimodal.data.tabular_dataset import resolve_isic2024_dataset_root
from isic2024_multimodal.features.tabular_terms import normalize_feature_set_name


FORBIDDEN_ORDINARY_COLUMNS = {
    "iddx_full",
    "diagnosis",
    "diagnosis_text",
    "pathology_text",
    "oracle_diagnosis",
    "target_derived_diagnosis",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    path: Path
    dataset_root: Path
    processed_dataset_root: Path | None
    feature_set_json: Path | None
    feature_sets: list[str]
    holdout_split_csv: Path
    cv_split_csv: Path
    cv_fold: int
    forbidden_ordinary_columns: set[str]
    payload: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_spec_path": str(self.path),
            "dataset_root": str(self.dataset_root),
            "processed_dataset_root": str(self.processed_dataset_root) if self.processed_dataset_root else None,
            "feature_set_json": str(self.feature_set_json) if self.feature_set_json else None,
            "feature_sets": self.feature_sets,
            "holdout_split_csv": str(self.holdout_split_csv),
            "cv_split_csv": str(self.cv_split_csv),
            "cv_fold": self.cv_fold,
            "forbidden_ordinary_columns": sorted(self.forbidden_ordinary_columns),
        }


def load_dataset_spec(path: str | Path, *, repo_root: str | Path | None = None) -> DatasetSpec:
    repo_root = Path(repo_root or Path.cwd()).resolve()
    spec_path = resolve_path(path, repo_root=repo_root)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError(f"Dataset spec is missing `dataset_id`: {spec_path}")

    feature_sets = [
        normalize_feature_set_name(name)
        for name in payload.get("feature_sets", payload.get("ordinary_feature_sets", []))
    ]
    forbidden = set(FORBIDDEN_ORDINARY_COLUMNS)
    forbidden.update(str(column) for column in payload.get("forbidden_ordinary_columns", []))

    spec = DatasetSpec(
        dataset_id=dataset_id,
        path=spec_path,
        dataset_root=resolve_path(payload.get("dataset_root", "data/raw"), repo_root=repo_root),
        processed_dataset_root=(
            resolve_path(payload["processed_dataset_root"], repo_root=repo_root)
            if payload.get("processed_dataset_root")
            else None
        ),
        feature_set_json=(
            resolve_path(payload["feature_set_json"], repo_root=repo_root)
            if payload.get("feature_set_json")
            else None
        ),
        feature_sets=feature_sets,
        holdout_split_csv=resolve_path(
            payload.get("holdout_split_csv", "data/splits/isic2024_train_validation_test_split_seed42.csv"),
            repo_root=repo_root,
        ),
        cv_split_csv=resolve_path(
            payload.get("cv_split_csv", "data/splits/isic2024_train_validation_5fold_seed42.csv"),
            repo_root=repo_root,
        ),
        cv_fold=int(payload.get("cv_fold", 0)),
        forbidden_ordinary_columns=forbidden,
        payload=payload,
    )
    validate_dataset_spec(spec)
    return spec


def validate_dataset_spec(spec: DatasetSpec) -> None:
    ordinary_columns = set(str(column) for column in spec.payload.get("ordinary_columns", []))
    if spec.feature_set_json and spec.feature_set_json.exists():
        feature_payload = json.loads(spec.feature_set_json.read_text(encoding="utf-8"))
        feature_sets = {
            normalize_feature_set_name(name): columns
            for name, columns in feature_payload.get("feature_sets", {}).items()
        }
        selected_feature_sets = spec.feature_sets or sorted(feature_sets)
        for feature_set_name in selected_feature_sets:
            ordinary_columns.update(str(column) for column in feature_sets.get(feature_set_name, []))

    forbidden_hits = sorted(ordinary_columns & spec.forbidden_ordinary_columns)
    if forbidden_hits:
        raise ValueError(
            "Dataset spec ordinary inference columns include privileged/diagnosis fields: "
            f"{forbidden_hits}"
        )


def dataset_fingerprint(spec: DatasetSpec) -> dict[str, Any]:
    paths = resolve_isic2024_dataset_root(spec.dataset_root, require_image_dir=False)
    metadata_path = paths.metadata_csv or paths.ground_truth_csv or paths.legacy_metadata_csv
    if metadata_path is None:
        raise RuntimeError(f"Could not identify metadata file for dataset root: {spec.dataset_root}")
    stat = metadata_path.stat()
    return {
        "dataset_root": str(paths.dataset_root),
        "dataset_format": paths.dataset_format,
        "metadata_path": str(metadata_path),
        "metadata_size_bytes": int(stat.st_size),
        "metadata_mtime_ns": int(stat.st_mtime_ns),
        "metadata_sha256": sha256_file(metadata_path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_root / value

