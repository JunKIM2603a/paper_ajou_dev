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
    if backend == "torchxrayvision":
        return _build_torchxrayvision_model(model_config, num_classes)
    if backend == "medclip":
        return _build_medclip_model(model_config, num_classes)
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
    elif architecture == "efficientnet_b3":
        model = models.efficientnet_b3(**kwargs)
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


class TorchXRayVisionEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_xrv_input(x)
        features = self.backbone.features(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def _to_xrv_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Expected image tensor with shape [batch, channels, height, width], got {tuple(x.shape)}")
        if x.shape[1] == 3:
            mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
            # Current ISIC pipeline yields ImageNet-normalized RGB images.
            # TorchXRayVision expects single-channel images with pixel scaling in [-1024, 1024].
            x = x * std + mean
            x = x.mean(dim=1, keepdim=True)
            x = x.clamp(0.0, 1.0)
            x = x * 2048.0 - 1024.0
            return x
        if x.shape[1] == 1:
            return x
        raise RuntimeError(f"TorchXRayVision backend expects 1 or 3 input channels, got {x.shape[1]}")


# TorchXRayVision 분기는 xrv backbone 위에 2-class classifier head를 얹어 transfer learning에 사용한다.
def _build_torchxrayvision_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "torchxrayvision is required for this model backend. Install it with `pip install torchxrayvision`."
        ) from exc

    architecture = str(model_config.get("architecture", "DenseNet")).lower()
    weights = model_config.get("weights")
    checkpoint_path = model_config.get("checkpoint_path")
    cache_dir = model_config.get("cache_dir")
    apply_sigmoid = bool(model_config.get("apply_sigmoid", False))

    builder_kwargs: dict[str, Any] = {
        "weights": weights,
        "apply_sigmoid": apply_sigmoid,
    }
    if cache_dir:
        builder_kwargs["cache_dir"] = str(cache_dir)

    if architecture in {"densenet", "densenet121"}:
        backbone = xrv.models.DenseNet(**builder_kwargs)
    elif architecture in {"resnet", "resnet50"}:
        backbone = xrv.models.ResNet(**builder_kwargs)
    else:
        raise ValueError(f"Unsupported torchxrayvision architecture: {architecture}")

    encoder = TorchXRayVisionEncoder(backbone)
    feature_dim = _infer_torchxrayvision_feature_dim(encoder, int(model_config.get("image_size", 224)))
    classifier = EncoderClassifier(encoder, feature_dim, num_classes)
    if checkpoint_path:
        _load_checkpoint(classifier, checkpoint_path)
    return classifier


class MedCLIPVisionEncoder(nn.Module):
    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.medclip_mean = 0.5862785803043838
        self.medclip_std = 0.27950088968644304

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_medclip_input(x)
        return self.vision_model(x)

    def _to_medclip_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Expected image tensor with shape [batch, channels, height, width], got {tuple(x.shape)}")
        if x.shape[1] == 3:
            mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
            x = x * std + mean
            x = x.mean(dim=1, keepdim=True)
        elif x.shape[1] != 1:
            raise RuntimeError(f"MedCLIP backend expects 1 or 3 input channels, got {x.shape[1]}")
        x = x.clamp(0.0, 1.0)
        x = (x - self.medclip_mean) / self.medclip_std
        return x


# 공식 MedCLIP package의 vision tower를 불러와 2-class classifier head를 얹는다.
def _build_medclip_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    try:
        from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPVisionModelViT
    except ImportError as exc:
        raise ImportError(
            "medclip is required for this model backend. Install it with `pip install medclip`."
        ) from exc

    architecture = str(model_config.get("architecture", "vit")).lower()
    checkpoint_path = model_config.get("checkpoint_path")
    pretrained = bool(model_config.get("pretrained", True))
    pretrained_dir = model_config.get("pretrained_dir")

    if architecture in {"vit", "vit-base", "clip-vit"}:
        vision_cls = MedCLIPVisionModelViT
    elif architecture in {"resnet", "resnet50"}:
        vision_cls = MedCLIPVisionModel
    else:
        raise ValueError(f"Unsupported MedCLIP architecture: {architecture}")

    medclip_model = MedCLIPModel(vision_cls=vision_cls)
    if pretrained:
        medclip_model.from_pretrained(input_dir=pretrained_dir)

    encoder = MedCLIPVisionEncoder(medclip_model.vision_model)
    feature_dim = _infer_medclip_feature_dim(encoder, int(model_config.get("image_size", 224)))
    classifier = EncoderClassifier(encoder, feature_dim, num_classes)
    if checkpoint_path:
        _load_checkpoint(classifier, checkpoint_path)
    return classifier


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


def _infer_torchxrayvision_feature_dim(encoder: nn.Module, image_size: int) -> int:
    with torch.no_grad():
        sample = torch.zeros(1, 3, image_size, image_size)
        features = encoder(sample)
    if features.ndim != 2:
        raise RuntimeError(f"Unexpected torchxrayvision feature shape: {tuple(features.shape)}")
    return int(features.shape[-1])


def _infer_medclip_feature_dim(encoder: nn.Module, image_size: int) -> int:
    with torch.no_grad():
        sample = torch.zeros(1, 3, image_size, image_size)
        features = encoder(sample)
    if features.ndim != 2:
        raise RuntimeError(f"Unexpected MedCLIP feature shape: {tuple(features.shape)}")
    return int(features.shape[-1])


