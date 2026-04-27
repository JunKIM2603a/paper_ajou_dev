# Hybrid Ensemble of Segmentation-Assisted Classification and GBDT

- Citation: Hasan and Rifat, 2025, arXiv:2506.03420, https://doi.org/10.48550/arXiv.2506.03420
- Publication type: preprint
- Seed citation: direct SLICE-3D use; exact citation relation 확인 필요
- Dataset: ISIC 2024 SLICE-3D plus external harmonized data
- Task: malignant vs benign skin lesion classification
- Modalities: image, engineered metadata, patient-specific relational metrics
- Inference inputs: image + engineered metadata; diagnosis-informed external relabeling used during training/data preparation
- Strict-contract compatibility: partially compatible

## Goal & Contribution
- Proposes a hybrid machine/deep learning system for ISIC 2024 non-dermoscopic 3D-TBP images.
- Combines segmentation-assisted image classification with GBDT metadata ensembles.
- Adds synthetic malignant lesions and diagnosis-informed harmonization of external datasets.

## Imbalance Handling
- Augments malignant cases with Stable Diffusion-generated synthetic lesions.
- Uses diagnosis-informed relabeling to map external data into a simplified three-class setup.
- This relabeling is useful but should be treated as candidate/external-data logic, not strict baseline.

## Models
- Tabular: GBDT ensemble with engineered features and patient relational metrics.
- Image: EVA02 and EdgeNeXtSAC.
- Fusion: image predictions are fused with metadata features in the GBDT ensemble.

## Metrics & Results
- Metrics: pAUC above 80% TPR.
- Best reported result: pAUC 0.1755.
- Validation/test protocol: paper says highest among configurations; exact split details 확인 필요.
- Threshold selection: not specified in abstract; 확인 필요.

## Limitations
- Preprint without full peer-review signal.
- Diagnosis-informed relabeling can conflict with strict inference/data-contract framing if not isolated.
- Synthetic data effects need separate ablation before being used as a paper claim.

## Relevance to Our Study
- Useful reference for hybrid fusion, metadata engineering, and synthetic positive augmentation.
- Supports designing a separate imbalance/synthetic-data ablation after strict baselines are stable.
- Not a clean ordinary-tabular baseline because of diagnosis-informed data preparation.

## Verification Notes
- Source: arXiv abstract.
