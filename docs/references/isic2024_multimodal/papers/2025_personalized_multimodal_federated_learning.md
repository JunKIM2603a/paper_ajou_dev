# Personalized Multimodal Federated Learning for Skin Cancer Diagnosis

- Citation: Fan et al., 2025, `Electronics`, https://doi.org/10.3390/electronics14142880
- Publication type: peer-reviewed
- Seed citation: ISIC 2024 challenge citation; exact seed citation 확인 필요
- Dataset: custom ISIC 2018-2024 mixture
- Task: skin cancer diagnosis under federated, heterogeneous, and missing-modality settings
- Modalities: image and tabular metadata
- Inference inputs: client-dependent image/tabular availability
- Strict-contract compatibility: partially compatible

## Goal & Contribution
- Proposes PMM-FL for personalized multimodal federated learning across heterogeneous clients.
- Targets missing modalities and privacy-preserving cross-institutional training.
- Uses multitask learning to combine diagnosis with missing-tabular-modality prediction.

## Imbalance Handling
- Builds a custom dataset by combining ISIC years and selecting all ISIC 2024 malignant examples.
- Adds positives from prior ISIC years and sampled benign examples to reduce imbalance.
- The paper text appears to report an ISIC 2024 count inconsistently; verify before citing.

## Models
- Tabular: tabular encoder and missing-modality prediction module.
- Image: CNN image encoder.
- Fusion: concatenated image/tabular features followed by multi-head attention and classifier.

## Metrics & Results
- Metrics: diagnostic accuracy, missing-modality robustness, communication overhead.
- Best reported result: 92.32% diagnostic accuracy; 2% drop under 30% modality missingness.
- Validation/test protocol: federated simulation; exact patient split 확인 필요.
- Threshold selection: not central / 확인 필요.

## Limitations
- Not a direct ISIC 2024 pAUC baseline because it mixes datasets and optimizes federated learning.
- Accuracy is less informative than pAUC/AP under ultra-rare malignant detection.
- Custom balancing makes direct comparison with strict ISIC 2024 protocol inappropriate.

## Relevance to Our Study
- Useful related work for missing metadata and heterogeneous deployment.
- Can inform future robustness experiments, not the first paper-facing baseline.
- Should be cited separately from strict image+tabular ISIC 2024 baselines.

## Verification Notes
- Source: MDPI article page and abstract snippets.
