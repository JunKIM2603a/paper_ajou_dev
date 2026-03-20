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

    model = timm.create_model(
        model_config["architecture"],
        pretrained=bool(model_config.get("pretrained", False)),
        num_classes=num_classes,
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
        _load_checkpoint(classifier, checkpoint_path)
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


# 외부 체크포인트는 module. 접두사가 붙어 있어도 최대한 유연하게 읽는다.
def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[checkpoint] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")








