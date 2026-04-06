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


# config.json의 backend 값에 따라 서로 다른 모델 생성 경로로 분기한다.
def build_model(model_config: dict[str, Any], num_classes: int = 2) -> nn.Module:
    backend = model_config["backend"]

    if backend == "torchvision":
        return _build_torchvision_model(model_config, num_classes)
    if backend == "timm":
        return _build_timm_model(model_config, num_classes)
    if backend == "open_clip":
        return _build_open_clip_model(model_config, num_classes)
    if backend == "huggingface_clip":
        return _build_huggingface_clip_model(model_config, num_classes)

    raise ValueError(f"Unsupported backend: {backend}")


# torchvision 계열은 마지막 분류 헤드만 2-class에 맞게 교체한다.
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
        _load_checkpoint(model, checkpoint_path)
    return model


# DeiT, DINOv2, RETFound 같은 timm 계열 모델을 생성한다.
def _build_timm_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for this model backend.") from exc

    image_size = model_config.get("image_size")
    model_kwargs: dict[str, Any] = {}
    if image_size is not None:
        model_kwargs["img_size"] = int(image_size)
    global_pool = model_config.get("global_pool")
    if global_pool:
        model_kwargs["global_pool"] = global_pool

    model = timm.create_model(
        model_config["architecture"],
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=num_classes,
        **model_kwargs,
    )
    checkpoint_path = model_config.get("checkpoint_path")
    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)
    return model


# CLIP류 모델은 이미지 인코더 출력 위에 별도 분류기를 얹어 fine-tuning 한다.
def _build_open_clip_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip_torch is required for this model backend.") from exc

    model_name = model_config["model_name"]
    pretrained = model_config.get("pretrained")
    checkpoint_path = model_config.get("checkpoint_path")
    if isinstance(pretrained, str) and pretrained.startswith(("hf-hub:", "local-dir:")):
        # open_clip 3.x에서는 hf/local schema를 pretrained가 아니라 model_name으로 받아들인다.
        model = open_clip.create_model_from_pretrained(
            pretrained,
            return_transform=False,
        )
    else:
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
    feature_dim = _infer_open_clip_feature_dim(model)
    classifier = EncoderClassifier(encoder, feature_dim, num_classes)
    if checkpoint_path:
        _load_open_clip_checkpoint(classifier, checkpoint_path)
    return classifier


# transformers 기반 비전 인코더도 feature extractor + classifier 구조로 감싼다.
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
            outputs = self.model.get_image_features(pixel_values=x)
            return outputs

    hidden_size = int(model_config["feature_dim"])
    classifier = EncoderClassifier(HuggingFaceVisionEncoder(backbone), hidden_size, num_classes)
    if checkpoint_path:
        _load_checkpoint(classifier, checkpoint_path)
    return classifier


# 외부 체크포인트는 module./model. 접두사나 head shape mismatch가 있어도 최대한 유연하게 읽는다.
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


def _load_open_clip_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    cleaned = _extract_checkpoint_state(checkpoint_path)

    # Some CheXzero/open_clip checkpoints store the raw CLIP model state_dict without the
    # EncoderClassifier wrapper prefix. When that happens, map them into encoder.clip_model.*
    # and leave the task-specific classifier head randomly initialized.
    if cleaned and not any(key.startswith(("encoder.clip_model.", "classifier.")) for key in cleaned):
        cleaned = {f"encoder.clip_model.{key}": value for key, value in cleaned.items()}

    filtered, skipped_unexpected, skipped_incompatible = _filter_compatible_state(model, cleaned)
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
    model: nn.Module, state: dict[str, torch.Tensor]
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


def _infer_open_clip_feature_dim(model: nn.Module) -> int:
    visual = getattr(model, "visual", None)
    if visual is not None:
        output_dim = getattr(visual, "output_dim", None)
        if isinstance(output_dim, int):
            return output_dim

        head = getattr(visual, "head", None)
        proj = getattr(head, "proj", None)
        out_features = getattr(proj, "out_features", None)
        if isinstance(out_features, int):
            return out_features

    with torch.no_grad():
        sample = torch.zeros(1, 3, 224, 224)
        features = model.encode_image(sample)
    if features.ndim != 2:
        raise RuntimeError(f"Unexpected open_clip image feature shape: {tuple(features.shape)}")
    return int(features.shape[-1])





