# CBIS-DDSM Model Benchmark

CBIS-DDSM binary classification benchmark for multiple pretrained backbones.
The task is binary classification on CBIS-DDSM:
`BENIGN` / `BENIGN_WITHOUT_CALLBACK` vs `MALIGNANT`.

## What Changed

- Every model config now uses `model.transfer_strategy = "full_finetune"` by default.
- You can override transfer learning mode from CLI with `--transfer-strategy full_finetune|linear_probe`.
- Relative `checkpoint_path` values are resolved relative to the config file directory.
- External checkpoint example configs were added for `CheXzero` and `RETFound`.
- A checkpoint guide was added: [docs/checkpoint_guide.md](docs/checkpoint_guide.md)

## Project Layout

- `cbis_ddsm_benchmark/`
- `1st_after/<model>/config.json`
- `1st_after/CheXzero/config.external_checkpoint.example.json`
- `1st_after/RETFound/config.external_checkpoint.example.json`
- `run_all_models.py`

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run One Model

```powershell
python -m cbis_ddsm_benchmark.run_experiment `
  --config .\1st_after\ResNet-50\config.json `
  --dataset-root .\dataset\archive_CBIS-DDSM_kaggle `
  --output-root .\artifacts
```

## Override Transfer Strategy

Full fine-tuning:

```powershell
python -m cbis_ddsm_benchmark.run_experiment `
  --config .\1st_after\ResNet-50\config.json `
  --transfer-strategy full_finetune
```

Linear probe:

```powershell
python -m cbis_ddsm_benchmark.run_experiment `
  --config .\1st_after\ResNet-50\config.json `
  --transfer-strategy linear_probe
```

## Run All Models

```powershell
python .\run_all_models.py `
  --dataset-root .\dataset\archive_CBIS-DDSM_kaggle `
  --output-root .\artifacts
```

Run all models with linear probe:

```powershell
python .\run_all_models.py `
  --dataset-root .\dataset\archive_CBIS-DDSM_kaggle `
  --output-root .\artifacts `
  --transfer-strategy linear_probe
```

## MLflow Report

```powershell
python -m cbis_ddsm_benchmark.mlflow_report
```

## Checkpoint Policy

The repository currently does not include tracked checkpoint files such as `.pt`, `.pth`, `.bin`, or `.safetensors`.
If a model needs a domain-specific checkpoint, you need to place it yourself and set `model.checkpoint_path`.

`checkpoint_path` rules:

- Absolute paths are used as-is.
- Relative paths are resolved from the folder that contains the `config.json` file.
- `state_dict`, `model_state_dict`, and `model` checkpoint wrappers are supported.
- `strict_checkpoint: false` allows partial loading.

## Model Status Summary

Verified in this repository on 2026-03-27.

| Model | Current init source in repo | Transfer status | External checkpoint needed later? | Notes |
|---|---|---|---|---|
| BioMedCLIP | Built-in `hf-hub` pretrained | `full_finetune` ready | No | Current config already points to a pretrained BioMedCLIP source |
| CheXzero | No built-in pretrained source configured | `full_finetune` ready | Yes | Add external checkpoint to use actual CheXzero weights |
| DeiT-S | `timm pretrained=true` | `full_finetune` ready | No | Built-in pretrained available |
| DenseNet-121 | `torchvision DEFAULT` | `full_finetune` ready | No | Built-in ImageNet weights |
| DINOv2 ViT-S | `timm pretrained=true` | `full_finetune` ready | No | Built-in pretrained available |
| EfficientNet-B0 | `torchvision DEFAULT` | `full_finetune` ready | No | Built-in ImageNet weights |
| EyePACS | `torchvision DEFAULT` | `full_finetune` ready | Optional | Current config uses generic ImageNet weights |
| HAM10000 | `torchvision DEFAULT` | `full_finetune` ready | Optional | Current config uses generic ImageNet weights |
| MedCLIP | `openai/clip-vit-base-patch16` | `full_finetune` ready | Optional | Works now, but not a MedCLIP-specific checkpoint |
| ResNet-50 | `torchvision DEFAULT` | `full_finetune` ready | No | Built-in ImageNet weights |
| RETFound | `pretrained=false` | `full_finetune` ready | Yes | Add external checkpoint to use actual RETFound weights |
| TorchXRayVision | `torchvision DEFAULT` | `full_finetune` ready | Optional | Current config is not using torchxrayvision-native weights |
| ViT-B_16 | `torchvision DEFAULT` | `full_finetune` ready | No | Built-in ImageNet weights |

## External Checkpoint Examples

CheXzero example:

- `1st_after/CheXzero/config.external_checkpoint.example.json`
- Example path: `../../external_checkpoints/CheXzero/chexzero_model.pt`

RETFound example:

- `1st_after/RETFound/config.external_checkpoint.example.json`
- Example path: `../../external_checkpoints/RETFound/retfound_model.pth`

## Notes

- `CheXzero` and `RETFound` are the two models that are most clearly checkpoint-dependent in the current repo state.
- If you later collect domain-specific weights for `EyePACS`, `HAM10000`, `MedCLIP`, or `TorchXRayVision`, you can reuse the same `checkpoint_path` mechanism.
