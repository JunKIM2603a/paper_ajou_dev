from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class CheckpointDownloadSpec:
    model_key: str
    target_path: str | None
    source_type: str
    source: str
    notes: str


CHECKPOINT_DOWNLOADS: dict[str, CheckpointDownloadSpec] = {
    "chexzero": CheckpointDownloadSpec(
        model_key="chexzero",
        target_path="checkpoints/CheXzero/best_128_5e-05_original_22000_0.855.pt",
        source_type="google_drive_folder",
        source="1makFLiEMbSleYltaRxw81aBhEDMpVwno",
        notes="CheXzero publishes weights in a Google Drive folder; this requires the optional gdown package.",
    ),
    "eyepacs": CheckpointDownloadSpec(
        model_key="eyepacs",
        target_path="checkpoints/EyePACS/eff_net_400x400.pt",
        source_type="url",
        source=(
            "https://raw.githubusercontent.com/skrsteski/diabetic-retinopathy-detection/"
            "main/results/models/eff_net_400x400.pt"
        ),
        notes="EfficientNet-B3 EyePACS checkpoint from the referenced GitHub repository.",
    ),
    "medclip": CheckpointDownloadSpec(
        model_key="medclip",
        target_path="checkpoints/MedCLIP/official-medclip-vit/pytorch_model.bin",
        source_type="zip_url",
        source="https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip",
        notes="Official MedCLIP-ViT weights downloaded by the MedCLIP project.",
    ),
    "retfound": CheckpointDownloadSpec(
        model_key="retfound",
        target_path="checkpoints/RETFound/RETFound_cfp_weights.pth",
        source_type="google_drive_file",
        source="1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE",
        notes="RETFound colour fundus ViT-Large checkpoint from the official model card.",
    ),
}


def download_for_model_config(
    model_config: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if model_config.get("checkpoint_path") is None and not (
        str(model_config.get("backend", "")).lower() == "medclip" and bool(model_config.get("pretrained", True))
    ):
        return {
            "model_key": infer_model_key(model_config),
            "status": "skipped",
            "reason": "config does not require a manual local checkpoint",
        }
    model_key = infer_model_key(model_config)
    return download_checkpoint(model_key, repo_root=repo_root, force=force)


def download_checkpoint(
    model_key: str,
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    normalized_key = model_key.lower()
    if normalized_key not in CHECKPOINT_DOWNLOADS:
        return {
            "model_key": normalized_key,
            "status": "skipped",
            "reason": "no manual checkpoint download is registered for this model",
        }

    spec = CHECKPOINT_DOWNLOADS[normalized_key]
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    target_path = resolve_repo_path(spec.target_path, repo_root=root) if spec.target_path else None
    if target_path is not None and target_path.exists() and not force:
        return {
            "model_key": normalized_key,
            "status": "exists",
            "path": str(target_path),
            "size_bytes": target_path.stat().st_size,
            "notes": spec.notes,
        }

    if target_path is None:
        raise RuntimeError(f"No target path is registered for {normalized_key}.")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if spec.source_type == "url":
        download_url(spec.source, target_path)
    elif spec.source_type == "zip_url":
        download_zip_member(spec.source, target_path)
    elif spec.source_type == "google_drive_file":
        download_google_drive_file(spec.source, target_path)
    elif spec.source_type == "google_drive_folder":
        download_google_drive_folder(spec.source, target_path)
    else:
        raise ValueError(f"Unsupported checkpoint source type for {normalized_key}: {spec.source_type}")

    if not target_path.exists():
        raise RuntimeError(f"Download for {normalized_key} completed but expected file is missing: {target_path}")
    return {
        "model_key": normalized_key,
        "status": "downloaded",
        "path": str(target_path),
        "size_bytes": target_path.stat().st_size,
        "notes": spec.notes,
    }


def infer_model_key(model_config: dict[str, Any]) -> str:
    display_name = str(model_config.get("display_name", "")).lower()
    architecture = str(model_config.get("architecture", "")).lower()
    checkpoint_path = str(model_config.get("checkpoint_path", "")).lower()
    if "chexzero" in display_name or "chexzero" in checkpoint_path:
        return "chexzero"
    if "eyepacs" in display_name or "eyepacs" in checkpoint_path:
        return "eyepacs"
    if "medclip" in display_name or str(model_config.get("backend", "")).lower() == "medclip":
        return "medclip"
    if "retfound" in display_name or "retfound" in architecture or "retfound" in checkpoint_path:
        return "retfound"
    return display_name.replace(" ", "_").replace("-", "_")


def resolve_repo_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return (root / raw_path).resolve()


def download_url(url: str, target_path: Path) -> None:
    temp_path = target_path.with_suffix(target_path.suffix + ".download")
    try:
        urlretrieve(url, temp_path)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def download_zip_member(url: str, target_path: Path) -> None:
    temp_zip = target_path.parent / "download.zip"
    try:
        urlretrieve(url, temp_zip)
        with zipfile.ZipFile(temp_zip) as archive:
            member_name = find_zip_member(archive, target_path.name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


def find_zip_member(archive: zipfile.ZipFile, filename: str) -> str:
    for member in archive.namelist():
        if Path(member).name == filename:
            return member
    raise RuntimeError(f"Zip archive does not contain {filename}.")


def download_google_drive_file(file_id: str, target_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Google Drive checkpoint download requires `gdown`. Install dependencies with `pip install gdown` "
            "or download the file manually."
        ) from exc

    gdown.download(id=file_id, output=str(target_path), quiet=False, fuzzy=False)


def download_google_drive_folder(folder_id: str, target_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Google Drive folder checkpoint download requires `gdown`. Install dependencies with `pip install gdown` "
            "or download the file manually."
        ) from exc

    download_dir = target_path.parent
    gdown.download_folder(id=folder_id, output=str(download_dir), quiet=False, use_cookies=False)
    if target_path.exists():
        return
    candidates = list(download_dir.rglob(target_path.name))
    if not candidates:
        raise RuntimeError(f"Downloaded Google Drive folder but did not find expected file: {target_path.name}")
    candidates[0].replace(target_path)
