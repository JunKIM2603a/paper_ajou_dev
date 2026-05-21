from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


SUPPORTED_BACKENDS = {"torchvision", "timm"}


# Image-only baselines are ordinary visual fine-tuning models. Dermatology VLMs
# such as MONET are reserved for future multimodal baselines.
def build_model(model_config: dict[str, Any], num_classes: int = 1) -> nn.Module:
    backend = str(model_config["backend"]).lower()

    if backend == "torchvision":
        return _build_torchvision_model(model_config, num_classes)
    if backend == "timm":
        return _build_timm_model(model_config, num_classes)

    raise ValueError(
        f"Unsupported image-only backend: {backend}. "
        f"Supported backends are: {sorted(SUPPORTED_BACKENDS)}"
    )


def _build_torchvision_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    architecture = str(model_config["architecture"])
    weights_name = model_config.get("weights")
    checkpoint_path = model_config.get("checkpoint_path")
    kwargs = {"weights": weights_name} if weights_name else {"weights": None}

    if architecture == "resnet50":
        model = models.resnet50(**kwargs)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "vit_b_16":
        model = models.vit_b_16(**kwargs)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported torchvision image-only architecture: {architecture}")

    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)
    return model


def _build_timm_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for timm image-only baselines.") from exc

    architecture = _resolve_timm_architecture(model_config, timm)
    image_size = model_config.get("image_size")
    model_kwargs: dict[str, Any] = {}
    if image_size is not None:
        model_kwargs["img_size"] = int(image_size)
    global_pool = model_config.get("global_pool")
    if global_pool:
        model_kwargs["global_pool"] = global_pool

    try:
        model = timm.create_model(
            architecture,
            pretrained=bool(model_config.get("pretrained", False)),
            num_classes=num_classes,
            **model_kwargs,
        )
    except TypeError:
        if "img_size" not in model_kwargs:
            raise
        model_kwargs.pop("img_size")
        model = timm.create_model(
            architecture,
            pretrained=bool(model_config.get("pretrained", False)),
            num_classes=num_classes,
            **model_kwargs,
        )

    checkpoint_path = model_config.get("checkpoint_path")
    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)
    return model


def _resolve_timm_architecture(model_config: dict[str, Any], timm_module: Any) -> str:
    candidates = model_config.get("architecture_candidates")
    if candidates:
        available = set(timm_module.list_models())
        for architecture in candidates:
            architecture = str(architecture)
            if architecture in available:
                return architecture
        raise ValueError(
            "None of the configured timm architecture_candidates are available: "
            f"{list(candidates)}"
        )
    return str(model_config["architecture"])


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    state = _extract_checkpoint_state(checkpoint_path)
    filtered, skipped_unexpected, skipped_incompatible = _filter_compatible_state(model, state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    unexpected = skipped_unexpected + list(unexpected)
    if missing:
        print(f"[checkpoint] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if skipped_incompatible:
        preview = [item[0] for item in skipped_incompatible[:5]]
        print(f"[checkpoint] Skipped incompatible keys: {preview}{'...' if len(skipped_incompatible) > 5 else ''}")


def _extract_checkpoint_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }


def _filter_compatible_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    target_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_unexpected: list[str] = []
    skipped_incompatible: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, value in state.items():
        if key not in target_state:
            skipped_unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_incompatible.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        filtered[key] = value

    return filtered, skipped_unexpected, skipped_incompatible
