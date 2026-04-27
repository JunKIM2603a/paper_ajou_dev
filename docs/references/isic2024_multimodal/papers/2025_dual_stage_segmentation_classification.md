# Dual-Stage Segmentation and Classification Framework

- Citation: Manzoor et al., 2025, `Digital Health`, https://doi.org/10.1177/20552076251351858
- Publication type: peer-reviewed
- Seed citation: SLICE-3D use; exact seed citation 확인 필요
- Dataset: HAM10000, ISIC 2018, ISIC 2024 SLICE-3D
- Task: skin lesion segmentation and classification
- Modalities: image and metadata for the SLICE-3D branch
- Inference inputs: image + metadata for fusion model
- Strict-contract compatibility: compatible / 확인 필요

## Goal & Contribution
- Proposes a two-phase framework: lesion segmentation followed by classification.
- Evaluates segmentation/classification across HAM10000, ISIC 2018, and SLICE-3D.
- Reports both tabular-only and image+metadata fusion results on ISIC 2024 SLICE-3D.

## Imbalance Handling
- Explicitly frames the method around balanced and imbalanced datasets.
- Uses SLICE-3D as a highly imbalanced and clinically realistic benchmark.
- Exact sampling, fold design, and class-weight strategy require PDF-level confirmation.

## Models
- Tabular: XGBoost classifier.
- Image: U-Net with VGG16 encoder for segmentation; ResNet-based classifier for SLICE fusion.
- Fusion: image + metadata fusion with a ResNet-based classifier.

## Metrics & Results
- Metrics: pAUC for SLICE-3D; accuracy, F1, sensitivity, specificity, Jaccard, Dice elsewhere.
- Best reported SLICE result: tabular-only XGBoost pAUC 0.16752.
- Image+tabular result: pAUC 0.15792 using ResNet fusion.
- Validation/test protocol: 확인 필요.
- Threshold selection: 확인 필요.

## Limitations
- Fusion underperforms tabular-only on reported SLICE result, so architecture or protocol may be weak.
- Results across datasets use different metrics, making comparison easy to overstate.
- Patient-level split and train-only preprocessing need verification before paper citation.

## Relevance to Our Study
- Valuable contrast case: image+tabular fusion is not automatically better than tabular-only.
- Supports our planned requirement to compare image-only, tabular-only, and multimodal under the same folds.
- Useful warning against claiming multimodal benefit without fold-wise ablation.

## Verification Notes
- Source: SAGE/Digital Health abstract and result snippets.
