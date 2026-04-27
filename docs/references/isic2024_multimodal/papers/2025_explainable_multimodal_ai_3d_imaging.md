# Explainable Multimodal AI via 3D Imaging and Clinical Data

- Citation: Wang et al., 2025, `Scientific Reports`, https://doi.org/10.1038/s41598-025-33536-z
- Publication type: peer-reviewed
- Seed citation: ISIC 2024 / SLICE-3D context; exact seed citation 확인 필요
- Dataset: ISIC 2024, 1,075 patients
- Task: six-class skin-lesion classification and binary challenge-style benchmark
- Modalities: 3D TBP images and structured clinical/lesion features
- Inference inputs: image + 41 clinical/lesion-specific features
- Strict-contract compatibility: incompatible / feature-audit required

## Goal & Contribution
- Develops an explainable multimodal AI framework for skin-lesion risk prediction.
- Combines image-based CNN outputs with structured clinical data and interpretability tools.
- Uses SHAP and CAM to connect model behavior to clinical/lesion features.
- Reported feature importance includes `mel_thick_mm`, which appears pathology-derived.

## Imbalance Handling
- Uses targeted augmentation for non-nevus classes according to secondary/source review.
- Binary challenge-style benchmark is reported separately from six-class classification.
- The pFPR wording needs careful mapping before comparison with pAUC@TPR>=0.80.

## Models
- Tabular: clinical-only XGBoost and multinomial logistic-regression decision/scoring model.
- Image: CNN trained on 3D TBP images.
- Fusion: multimodal fusion model combining image and structured clinical features.

## Metrics & Results
- Metrics: accuracy, recall, F1, AUC, pFPR/challenge-style score.
- Best reported result: recall and F1 above 95%, AUC above 0.95; pFPR 0.1734 for challenge-style benchmark.
- Validation/test protocol: internal evaluation and ISIC 2024 challenge-style comparison; exact split 확인 필요.
- Threshold selection: 확인 필요.

## Limitations
- Six-class diagnostic framing is not the same as the ISIC 2024 malignant binary target.
- Reported `mel_thick_mm` is not an ordinary inference-time metadata feature and violates strict-contract comparison unless removed.
- pFPR is not directly interchangeable with the repository pAUC definition without verification.

## Relevance to Our Study
- Useful XAI reference for SHAP and CAM in image+tabular dermatology models.
- Supports including interpretability as secondary evidence, not as a substitute for leakage-safe metrics.
- Not directly comparable as a strict baseline unless pathology-derived features are removed and re-audited.

## Verification Notes
- Source: Scientific Reports article, PubMed/PMC abstract, and article discussion snippets.
