# Image Baseline Checkpoint Policy

The image-only baseline suite now uses only ordinary pretrained visual
backbones from `torchvision` and `timm`.

Active image-only models:

- `ResNet50`
- `EfficientNetV2-S`
- `ConvNeXtV2-Tiny`
- `EVA-02-S`
- `ViT-B`
- `EdgeNeXt-S`

Manual medical-image checkpoint download helpers were removed from the
image-only path. `MONET` and other image-text or multimodal encoders are
reserved for future multimodal baseline experiments.

Smoke or offline checks may still pass `--disable-pretrained`, which disables
library pretrained weights for the current run only.
