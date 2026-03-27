from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def build_model(model_config: dict[str, Any], num_classes: int = 2) -> nn.Module:
    backend = model_config["backend"]

    if backend == "torchvision":
        model = _build_torchvision_model(model_config, num_classes)
    elif backend == "timm":
        model = _build_timm_model(model_config, num_classes)
    elif backend == "open_clip":
        model = _build_open_clip_model(model_config, num_classes)
    elif backend == "huggingface_clip":
        model = _build_huggingface_clip_model(model_config, num_classes)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    _apply_transfer_learning_strategy(model, model_config)
    return model


def _build_torchvision_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    architecture = model_config["architecture"]
    weights_name = model_config.get("weights")
    checkpoint_path = model_config.get("checkpoint_path")
    kwargs = {"weights": weights_name} if weights_name else {"weights": None}

    if architecture == "resnet50":
        model = models.resnet50(**kwargs)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "densenet121":
        model = models.densenet121(**kwargs)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(**kwargs)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture == "vit_b_16":
        model = models.vit_b_16(**kwargs)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported torchvision architecture: {architecture}")

    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path, strict=bool(model_config.get("strict_checkpoint", False)))
    return model


def _build_timm_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for this model backend.") from exc

    model = timm.create_model(
        model_config["architecture"],
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=num_classes,
    )
    checkpoint_path = model_config.get("checkpoint_path")
    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path, strict=bool(model_config.get("strict_checkpoint", False)))
    return model


def _build_open_clip_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip_torch is required for this model backend.") from exc

    model_name = model_config["model_name"]
    pretrained = model_config.get("pretrained")
    checkpoint_path = model_config.get("checkpoint_path")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )

    class OpenClipVisionEncoder(nn.Module):
        def __init__(self, clip_model: nn.Module) -> None:
            super().__init__()
            self.clip_model = clip_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.clip_model.encode_image(x)

    encoder = OpenClipVisionEncoder(model)
    feature_dim = model.visual.output_dim
    classifier = EncoderClassifier(encoder, feature_dim, num_classes)
    if checkpoint_path:
        _load_checkpoint(classifier, checkpoint_path, strict=bool(model_config.get("strict_checkpoint", False)))
    return classifier


def _build_huggingface_clip_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("transformers is required for this model backend.") from exc

    backbone = AutoModel.from_pretrained(model_config["hf_model_id"])
    checkpoint_path = model_config.get("checkpoint_path")

    class HuggingFaceVisionEncoder(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.get_image_features(pixel_values=x)

    hidden_size = int(model_config["feature_dim"])
    classifier = EncoderClassifier(HuggingFaceVisionEncoder(backbone), hidden_size, num_classes)
    if checkpoint_path:
        _load_checkpoint(classifier, checkpoint_path, strict=bool(model_config.get("strict_checkpoint", False)))
    return classifier


def _load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = False) -> None:
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    state = torch.load(resolved_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"[checkpoint] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _apply_transfer_learning_strategy(model: nn.Module, model_config: dict[str, Any]) -> None:
    strategy = str(model_config.get("transfer_strategy", "full_finetune")).lower()
    head_patterns = tuple(model_config.get("head_patterns", ["classifier", "fc", "heads.head", "head"]))

    if strategy == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    if strategy in {"linear_probe", "head_only"}:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(pattern in name for pattern in head_patterns)
        return

    raise ValueError(f"Unsupported transfer strategy: {strategy}")
