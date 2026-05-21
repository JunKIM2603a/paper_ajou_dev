# Image Baseline Checkpoint Policy

현재 image-only baseline suite는 `torchvision`과 `timm`의 일반 pretrained visual backbone만 사용한다.

현재 image-only 모델:

- `ResNet50`
- `EfficientNetV2-S`
- `ConvNeXtV2-Tiny`
- `EVA-02-S`
- `ViT-B`
- `EdgeNeXt-S`

수동 medical-image checkpoint 다운로드 helper는 image-only 경로에서 제거했다. `MONET`과 기타 image-text 또는 multimodal encoder는 향후 multimodal baseline 실험용으로 남겨 둔다.

Smoke 또는 offline 점검에서는 `--disable-pretrained`를 사용할 수 있다. 이 옵션은 현재 실행에서만 library pretrained weight를 비활성화한다.
