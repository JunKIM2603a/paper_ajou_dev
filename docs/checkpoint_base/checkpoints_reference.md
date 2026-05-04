# Checkpoints Reference

This file records checkpoint expectations for ISIC2024 image-only baselines. As of this
workspace audit, `/home/ubuntu/wksp/paper_ajou_dev/checkpoints/` does not exist, so any
config that requires a local checkpoint will fail preflight until that file is added.

Run a config preflight with:

```bash
python -m isic2024_multimodal.cli.run_image_baseline --config experiments/configs/image_baselines/resnet50/config.json --preflight-only
```

Download registered checkpoints with:

```bash
python -m isic2024_multimodal.cli.download_image_checkpoints --models chexzero eyepacs medclip retfound
```

Or let an image run download the registered checkpoint before preflight:

```bash
python -m isic2024_multimodal.cli.run_image_baseline --config experiments/configs/image_baselines/retfound/config.json --auto-download-checkpoints --preflight-only
```

Run a smoke suite with:

```bash
python -m isic2024_multimodal.cli.run_all_image_models --models resnet50 --auto-download-checkpoints --max-trials 1 --epochs-override 1 --max-train-samples 256 --max-val-samples 128 --max-test-samples 128
```

Check tested / untested model status with:

```bash
python -m isic2024_multimodal.cli.image_baseline_status
```

The status command writes:

```text
experiments/tables/image_baselines/status/image_baseline_status.csv
experiments/tables/image_baselines/status/image_baseline_status.json
```

Google Drive based downloads require `gdown`, which is listed in `requirements.txt`.

## BioMedCLIP

- Config: `experiments/configs/image_baselines/biomedclip/config.json`
- Backend: `open_clip`
- Pretrained source: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Manual checkpoint required: no
- Current status: runnable only if `open_clip` can download or reuse the Hugging Face cache
- Appropriateness: medical image-text pretraining baseline, not skin-lesion-specific

## CheXzero

- Config: `experiments/configs/image_baselines/chexzero/config.json`
- Local file required: `checkpoints/CheXzero/best_128_5e-05_original_22000_0.855.pt`
- Repo: [CheXzero](https://github.com/rajpurkarlab/CheXzero)
- Weights source: [Google Drive folder](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)
- Current status: missing in this workspace; use `download_image_checkpoints --models chexzero`
- Appropriateness: chest X-ray pretrained, out-of-domain medical-transfer baseline

## DeiT-S

- Config: `experiments/configs/image_baselines/deit_s/config.json`
- Backend: `timm`
- Pretrained source: timm pretrained weights/cache
- Manual checkpoint required: no
- Appropriateness: general-vision baseline

## DenseNet-121

- Config: `experiments/configs/image_baselines/densenet121/config.json`
- Backend: `torchvision`
- Pretrained source: torchvision DEFAULT weights/cache
- Manual checkpoint required: no
- Appropriateness: general-vision baseline

## DINOv2 ViT-S

- Config: `experiments/configs/image_baselines/dinov2_vit_s/config.json`
- Backend: `timm`
- Pretrained source: timm pretrained weights/cache
- Manual checkpoint required: no
- Appropriateness: general self-supervised vision baseline

## EfficientNet-B0

- Config: `experiments/configs/image_baselines/efficientnet_b0/config.json`
- Backend: `torchvision`
- Pretrained source: torchvision DEFAULT weights/cache
- Manual checkpoint required: no
- Appropriateness: general-vision baseline

## EyePACS

- Config: `experiments/configs/image_baselines/eyepacs/config.json`
- Local file required: `checkpoints/EyePACS/eff_net_400x400.pt`
- Reference: [diabetic-retinopathy-detection checkpoint](https://github.com/skrsteski/diabetic-retinopathy-detection/blob/main/results/models/eff_net_400x400.pt)
- Expected structure: `torchvision efficientnet_b3` backbone with the original 5-class classifier head skipped
- Current status: missing in this workspace; use `download_image_checkpoints --models eyepacs`
- Appropriateness: retinal/fundus pretrained, out-of-domain medical-transfer baseline

## MedCLIP

- Config: `experiments/configs/image_baselines/medclip/config.json`
- Official classification repo: [RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)
- Required official local weights: `checkpoints/MedCLIP/official-medclip-vit/pytorch_model.bin`
- Not suitable for this baseline: `checkpoints/MedCLIP/sam_vit_b_01ec64.pth`
- Current status: official MedCLIP-ViT weights are missing in this workspace; use `download_image_checkpoints --models medclip`
- Appropriateness: medical image-text pretraining baseline, not skin-lesion-specific

`sam_vit_b_01ec64.pth` is a SAM ViT-B checkpoint. It is not the official MedCLIP
image-text checkpoint and must not be used as the MedCLIP classifier baseline.

## ResNet-50

- Config: `experiments/configs/image_baselines/resnet50/config.json`
- Backend: `torchvision`
- Pretrained source: torchvision DEFAULT weights/cache
- Manual checkpoint required: no
- Appropriateness: general-vision baseline

## RETFound

- Config: `experiments/configs/image_baselines/retfound/config.json`
- Reference repo: [open-eye/RETFound_MAE](https://huggingface.co/open-eye/RETFound_MAE)
- Local file required: `checkpoints/RETFound/RETFound_cfp_weights.pth`
- Expected load pattern: apply `checkpoint["model"]` to the ViT backbone and skip incompatible head weights
- Current status: missing in this workspace; use `download_image_checkpoints --models retfound`
- Appropriateness: retinal/fundus pretrained, out-of-domain medical-transfer baseline

## TorchXRayVision

- Config: `experiments/configs/image_baselines/torchxrayvision/config.json`
- Backend: `torchxrayvision`
- Load method:

```python
import torchxrayvision as xrv
model = xrv.models.DenseNet(weights="densenet121-res224-all")
```

- Runtime cache dir: `checkpoints/TorchXRayVision`
- Manual checkpoint required: no, but the package must download or reuse cached weights
- Current status: no local cache directory exists in this workspace
- Appropriateness: chest X-ray pretrained, out-of-domain medical-transfer baseline

## ViT-B/16

- Config: `experiments/configs/image_baselines/vit_b_16/config.json`
- Backend: `torchvision`
- Pretrained source: torchvision DEFAULT weights/cache
- Manual checkpoint required: no
- Appropriateness: general-vision baseline

## Excluded From Requested Suite

- `ham10000`: config exists for reference but is excluded from `experiments/configs/suites/image_baselines.json`
- `monet`: config exists for reference but is excluded from `experiments/configs/suites/image_baselines.json`
