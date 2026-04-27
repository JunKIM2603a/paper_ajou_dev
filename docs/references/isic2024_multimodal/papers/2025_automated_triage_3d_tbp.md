# Automated Triage with 3D Total-Body Photography

- Citation: Kurtansky et al., 2025, `npj Digital Medicine`, https://doi.org/10.1038/s41746-025-02070-7
- Publication type: peer-reviewed
- Seed citation: SLICE-3D; this is also seed paper 2
- Dataset: ISIC 2024 / SLICE-3D training data and private leaderboard evaluation data
- Task: pathology-confirmed skin cancer triage from 3D-TBP lesion tiles
- Modalities: tile image, basic demographics/location, WB360 appearance metadata, patient context
- Inference inputs: image tiles + metadata + patient-contextual features
- Strict-contract compatibility: partially compatible

## Goal & Contribution
- Summarizes ISIC 2024 challenge outcomes and evaluates clinical plausibility of automated atypical lesion triage.
- Provides an ablation of tile images, demographic metadata, WB360 appearance metadata, and patient-contextual features.
- Shows that multimodal and patient-contextual information materially improves triage performance.

## Imbalance Handling
- Uses pAUC above 80% TPR as the primary leaderboard metric to emphasize low false-negative operation.
- Reports patient-level triage style metrics such as top-15 sensitivity and NNT at sensitivity thresholds.
- Extreme rare-positive setting remains a central limitation; leaderboard probing/overfitting is explicitly discussed.

## Models
- Tabular: gradient boosting tree models over metadata and patient-contextual features.
- Image: independent EVA / EdgeNeXt-style image models from the winning Kaggle solution.
- Fusion: late fusion; image model probability estimates plus metadata features feed GBT models.

## Metrics & Results
- Metrics: pAUC>80%TPR, full ROC-AUC, SE top-15, NNT at sensitivity thresholds.
- Best reported result: skin cancer pAUC 0.1726 and AUC 0.9668 on private leaderboard.
- Melanoma-specific result: pAUC 0.1757, AUC 0.9704, SE top-15 0.7908.
- Validation/test protocol: Kaggle hidden public/private leaderboard; ablation on private evaluation set.
- Threshold selection: challenge metric threshold-free; NNT evaluated at fixed sensitivity thresholds.

## Limitations
- Patient-contextual features require multiple lesions from the same patient.
- WB360 appearance metadata comes from proprietary tooling and is not always externally available.
- Public leaderboard overfitting is acknowledged, so private scores are more reliable.

## Relevance to Our Study
- Strongest reference for ISIC 2024 metric framing and multimodal feature-class ablation.
- Supports our need to compare image-only, tabular-only, and multimodal baselines under pAUC@TPR>=0.80.
- Not a pure single-lesion strict baseline because patient context and proprietary features can be unavailable.

## Verification Notes
- Source: Nature article, lines on pAUC metric, private scores, ablation, and feature classes.
