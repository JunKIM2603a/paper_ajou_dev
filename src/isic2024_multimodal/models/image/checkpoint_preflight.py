from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SUPPORTED_IMAGE_ONLY_BACKENDS = {"torchvision", "timm"}


def preflight_image_model_config(model_config: dict[str, Any], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    backend = str(model_config.get("backend", "")).lower()
    display_name = str(model_config.get("display_name", model_config.get("architecture", backend)))
    checkpoint_path = model_config.get("checkpoint_path")
    if backend not in SUPPORTED_IMAGE_ONLY_BACKENDS:
        raise ValueError(
            f"Unsupported image-only backend: {backend}. "
            f"Supported backends are: {sorted(SUPPORTED_IMAGE_ONLY_BACKENDS)}"
        )

    report: dict[str, Any] = {
        "model_name": display_name,
        "backend": backend,
        "checkpoint_path": checkpoint_path,
        "checkpoint_required": bool(checkpoint_path),
        "checkpoint_exists": None,
        "checkpoint_size_bytes": None,
        "expected_loader": expected_loader_name(model_config),
        "state_key_count": None,
        "state_key_preview": [],
        "status": "ok",
        "notes": [],
    }

    if checkpoint_path:
        resolved = resolve_repo_path(checkpoint_path, repo_root=root)
        report["resolved_checkpoint_path"] = str(resolved)
        report["checkpoint_exists"] = resolved.exists()
        if not resolved.exists():
            raise FileNotFoundError(
                f"{display_name} requires local checkpoint '{checkpoint_path}', but it was not found at {resolved}."
            )
        report["checkpoint_size_bytes"] = resolved.stat().st_size
        state = load_checkpoint_state(resolved)
        report.update(inspect_checkpoint_state(state))
        report.update(checkpoint_compatibility_report(model_config, state))
        if report["compatible_key_count"] == 0:
            raise ValueError(
                f"{display_name} checkpoint exists at {resolved}, but no compatible model keys were found."
            )
    else:
        report["notes"].append("uses library pretrained weights/cache; no manual local checkpoint required")

    return report


def resolve_repo_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return (root / raw_path).resolve()


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {path}")
    return state


def inspect_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(str(key).removeprefix("module.").removeprefix("model.") for key in state.keys())
    return {
        "state_key_count": len(keys),
        "state_key_preview": keys[:10],
    }


def checkpoint_compatibility_report(model_config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    from isic2024_multimodal.models.image.factory import build_model

    target_config = dict(model_config)
    target_config["checkpoint_path"] = None
    if "weights" in target_config:
        target_config["weights"] = None
    if "pretrained" in target_config:
        target_config["pretrained"] = False

    model = build_model(target_config, num_classes=1)
    target_state = model.state_dict()
    cleaned_state = clean_checkpoint_keys(state)

    compatible = 0
    incompatible = 0
    unexpected = 0
    for key, value in cleaned_state.items():
        if key not in target_state:
            unexpected += 1
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(target_state[key].shape):
            incompatible += 1
            continue
        compatible += 1

    return {
        "target_state_key_count": len(target_state),
        "compatible_key_count": compatible,
        "unexpected_key_count": unexpected,
        "incompatible_key_count": incompatible,
    }


def clean_checkpoint_keys(state: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key).removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }


def expected_loader_name(model_config: dict[str, Any]) -> str:
    backend = str(model_config.get("backend", "")).lower()
    if backend == "timm":
        return "timm state_dict checkpoint adapter"
    if backend == "torchvision":
        return "torchvision state_dict checkpoint adapter"
    return backend or "unknown"
