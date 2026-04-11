# Checkpoints Reference

## CheXzero

- Local file: `checkpoints/CheXzero/best_128_5e-05_original_22000_0.855.pt`
- Repo: [CheXzero](https://github.com/rajpurkarlab/CheXzero)
- Weights source: [Google Drive folder](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)
- Current status: runnable with the local checkpoint already present in this workspace

## RETFound

- Reference repo: [open-eye/RETFound_MAE](https://huggingface.co/open-eye/RETFound_MAE)
- Local file: `checkpoints/RETFound/RETFound_cfp_weights.pth`
- HF example pattern: load local `RETFound_cfp_weights.pth` and then apply `checkpoint['model']` to the ViT backbone
- Current status: runnable in this workspace with the local checkpoint already present

## EyePACS

- Local file: `checkpoints/EyePACS/eff_net_400x400.pt`
- Reference: [diabetic-retinopathy-detection checkpoint](https://github.com/skrsteski/diabetic-retinopathy-detection/blob/main/results/models/eff_net_400x400.pt)
- Local checkpoint structure matches `torchvision efficientnet_b3` except for the original `5-class` classifier head
- Current benchmark config uses `efficientnet_b3`, `image_size=400`, and the local checkpoint path above
- Current status: runnable for full fine-tuning in this workspace

## TorchXRayVision

- Load method:

```python
import torchxrayvision as xrv
model = xrv.models.DenseNet(weights="densenet121-res224-all")
```

- Install command: `conda run -n paper_ajou_dev python -m pip install torchxrayvision`
- Runtime cache dir: `checkpoints/TorchXRayVision`
- Current status: benchmark loader backend is wired and the package is installed in this workspace, so full fine-tuning is runnable

## HAM10000

- Excluded from the current follow-up benchmark pool by request
- Reference only: [EfficientNet-Skin-Cancer](https://github.com/atlan-antillia/EfficientNet-Skin-Cancer?tab=readme-ov-file)

## MedCLIP

- Local file: `checkpoints/MedCLIP/sam_vit_b_01ec64.pth`
- This file is a `SAM ViT-B` checkpoint, not the official MedCLIP image-text checkpoint
- Official classification repo: [RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)
- Install command: `conda run -n paper_ajou_dev python -m pip install medclip`
- Current benchmark config downloads the official MedCLIP-ViT weights into `checkpoints/MedCLIP/official-medclip-vit/pytorch_model.bin` on first use, then reuses that local cache
- `medclip-samv2` reference: [healthx-lab/medclip-samv2](https://github.com/healthx-lab/medclip-samv2?tab=readme-ov-file)
- If you want to reproduce `medclip-samv2`, the existing `sam_vit_b_01ec64.pth` is the SAM side only; you additionally need the repo's `pytorch_model.bin`
