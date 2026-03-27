# Checkpoint Guide

Verified on 2026-03-27.

This guide explains which models in this repository can run with built-in pretrained weights and which ones are better used with an external checkpoint.

## Path Rules

- `model.checkpoint_path` can be absolute.
- If it is relative, it is resolved from the folder that contains the `config.json` file.
- Example: `1st_after/CheXzero/config.external_checkpoint.example.json` uses `../../external_checkpoints/CheXzero/chexzero_model.pt`.
- Example: `1st_after/RETFound/config.external_checkpoint.example.json` uses `../../external_checkpoints/RETFound/retfound_model.pth`.

## Model Table

| Model | Repo default | External checkpoint later? | Recommended source | Why |
|---|---|---|---|---|
| BioMedCLIP | Built-in HF Hub source | No | Microsoft Hugging Face model card | Current config already points to the official pretrained source |
| CheXzero | No pretrained source configured | Yes | Rajpurkar Lab CheXzero repo checkpoints | Current config has `pretrained: null`, so actual CheXzero transfer learning needs external weights |
| DeiT-S | `timm pretrained=true` | No | Built-in `timm` weights | Works with built-in pretrained weights |
| DenseNet-121 | `torchvision DEFAULT` | No | Built-in `torchvision` weights | Works with built-in pretrained weights |
| DINOv2 ViT-S | `timm pretrained=true` | No | Built-in `timm` weights | Works with built-in pretrained weights |
| EfficientNet-B0 | `torchvision DEFAULT` | No | Built-in `torchvision` weights | Works with built-in pretrained weights |
| EyePACS | `torchvision DEFAULT` | Optional | Your own EyePACS pretrained checkpoint | Current repo uses generic ImageNet initialization |
| HAM10000 | `torchvision DEFAULT` | Optional | Your own HAM10000 pretrained checkpoint | Current repo uses generic ImageNet initialization |
| MedCLIP | `openai/clip-vit-base-patch16` | Optional | Official MedCLIP repo | Current repo works, but this is not a MedCLIP-specific checkpoint |
| ResNet-50 | `torchvision DEFAULT` | No | Built-in `torchvision` weights | Works with built-in pretrained weights |
| RETFound | `pretrained=false` | Yes | Official RETFound repo / Hugging Face access flow | Current config does not load RETFound weights unless you provide them |
| TorchXRayVision | `torchvision DEFAULT` | Optional | Official TorchXRayVision repo | Current repo name suggests XRV, but config is using plain torchvision DenseNet121 |
| ViT-B_16 | `torchvision DEFAULT` | No | Built-in `torchvision` weights | Works with built-in pretrained weights |

## Recommended Links

- BioMedCLIP: <https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224>
- CheXzero: <https://github.com/rajpurkarlab/CheXzero>
- MedCLIP: <https://github.com/RyanWangZf/MedCLIP>
- RETFound: <https://github.com/rmaphoh/RETFound_MAE>
- TorchXRayVision: <https://github.com/mlmed/torchxrayvision>

## Notes

- In the current repository state, the most clearly external-checkpoint-dependent models are `CheXzero` and `RETFound`.
- `EyePACS`, `HAM10000`, `MedCLIP`, and `TorchXRayVision` can still run transfer learning now, but only with generic or alternate pretrained sources unless you provide their domain-specific checkpoints.
- The repository does not currently contain tracked `.pt`, `.pth`, `.bin`, or `.safetensors` files.
