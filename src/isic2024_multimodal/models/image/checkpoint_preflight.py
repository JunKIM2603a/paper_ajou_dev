from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
MEDCLIP_SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"


def preflight_image_model_config(model_config: dict[str, Any], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    backend = str(model_config.get("backend", ""))
    display_name = str(model_config.get("display_name", model_config.get("architecture", backend)))
    checkpoint_path = model_config.get("checkpoint_path")

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
        if backend == "medclip" and resolved.name == MEDCLIP_SAM_CHECKPOINT_NAME:
            raise ValueError(
                "MedCLIP config points to sam_vit_b_01ec64.pth, which is a SAM checkpoint and is not usable "
                "as the official MedCLIP image-text checkpoint."
            )
        state = load_checkpoint_state(resolved)
        report.update(inspect_checkpoint_state(state))
        report.update(checkpoint_compatibility_report(model_config, state))
        if report["compatible_key_count"] == 0:
            raise ValueError(
                f"{display_name} checkpoint exists at {resolved}, but no compatible model keys were found."
            )

    if backend == "medclip" and bool(model_config.get("pretrained", True)):
        pretrained_dir = model_config.get("pretrained_dir")
        if pretrained_dir:
            resolved_dir = resolve_repo_path(pretrained_dir, repo_root=root)
            weights_file = resolved_dir / "pytorch_model.bin"
            report["resolved_pretrained_dir"] = str(resolved_dir)
            report["pretrained_weights_file"] = str(weights_file)
            report["pretrained_weights_exist"] = weights_file.exists()
            if not weights_file.exists():
                raise FileNotFoundError(
                    "MedCLIP official pretrained weights are required for this config. "
                    f"Expected {weights_file}. The SAM checkpoint is not a substitute."
                )

    if backend in {"torchvision", "timm"} and not checkpoint_path:
        report["notes"].append("uses library pretrained weights/cache; no manual local checkpoint required")
    if backend == "open_clip" and not checkpoint_path:
        report["notes"].append("uses open_clip pretrained source/cache; no manual local checkpoint required")
    if backend == "torchxrayvision" and not checkpoint_path:
        report["notes"].append("uses torchxrayvision package weights/cache; no manual local checkpoint required")

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
    backend = str(target_config.get("backend", ""))
    target_config["checkpoint_path"] = None
    if "weights" in target_config:
        target_config["weights"] = None
    if backend != "open_clip" and "pretrained" in target_config:
        target_config["pretrained"] = False

    model = build_model(target_config, num_classes=2)
    target_state = model.state_dict()
    cleaned_state = clean_checkpoint_keys(state)
    if backend == "open_clip" and cleaned_state and not any(
        key.startswith(("encoder.clip_model.", "classifier.")) for key in cleaned_state
    ):
        cleaned_state = {f"encoder.clip_model.{key}": value for key, value in cleaned_state.items()}

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
    backend = str(model_config.get("backend", ""))
    if backend == "open_clip":
        return "open_clip EncoderClassifier checkpoint adapter"
    if backend == "timm":
        return "timm state_dict checkpoint adapter"
    if backend == "torchvision":
        return "torchvision state_dict checkpoint adapter"
    if backend == "medclip":
        return "official MedCLIP pretrained directory"
    if backend == "torchxrayvision":
        return "torchxrayvision package weights/cache"
    if backend == "huggingface_clip":
        return "transformers AutoModel weights/cache"
    return backend or "unknown"
