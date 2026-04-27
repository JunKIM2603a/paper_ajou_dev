# Multimodal System for Skin Cancer Detection

- Citation: Sydorskyi et al., 2026, arXiv:2601.14822, https://doi.org/10.48550/arXiv.2601.14822
- Publication type: preprint; accepted to `System research and information technologies`
- Seed citation: SLICE-3D citation likely; exact cited-by relation 확인 필요
- Dataset: ISIC 2024, ISIC Archive, and generated data
- Task: melanoma / malignant lesion detection using conventional photo images
- Modalities: image and tabular metadata
- Inference inputs: image + tabular metadata; also supports cases without metadata
- Strict-contract compatibility: compatible / partially compatible

## Goal & Contribution
- Builds an accessible multimodal melanoma detection system using conventional, non-dermoscopic photos.
- Studies one-stage, two-stage, and three-stage pipelines for metadata-aware and metadata-missing settings.
- Provides ablations over vision architectures, boosting algorithms, and loss functions.

## Imbalance Handling
- Uses external ISIC Archive data and generated data to increase malignant examples.
- Evaluates loss functions and multi-stage refinement under extreme rare-positive conditions.
- Excludes ISIC 2024 patients from ISIC Archive data according to available source snippets.

## Models
- Tabular: XGBoost and LightGBM variants.
- Image: Multi-Modal ConvNeXt, EdgeNeXt, and other recent vision architectures.
- Fusion: multimodal neural network plus two-stage/three-stage boosted refinement.

## Metrics & Results
- Metrics: Partial ROC AUC, ROC AUC, Top-15 retrieval sensitivity.
- Best reported result: peak pAUC 0.18068 and top-15 retrieval sensitivity 0.78371.
- Validation/test protocol: 5-fold CV and Kaggle public/private benchmarks according to secondary review snippets.
- Threshold selection: not clearly specified; 확인 필요.

## Limitations
- Preprint status; details should be checked in PDF before strong paper claims.
- External/generated data make it not directly comparable to a strict ISIC-only baseline.
- Leaderboard benchmarks can be useful but should not be the only evidence.

## Relevance to Our Study
- Strong engineering reference for multimodal ConvNeXt/boosting and metadata-missing robustness.
- Useful for imbalance/loss/threshold ablation planning.
- We should separate its external/generative augmentation from strict baseline comparisons.

## Verification Notes
- Source: arXiv abstract and secondary review/source snippets.
