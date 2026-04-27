# FusionNetX

- Citation: Nguyen et al., 2025, `Journal of Computer Science and Cybernetics`, https://doi.org/10.15625/1813-9663/22005
- Publication type: peer-reviewed
- Seed citation: SLICE-3D / ISIC 2024 citation 확인 필요
- Dataset: ISIC 2024
- Task: skin cancer detection from 3D-TBP lesion crops
- Modalities: image data and metadata
- Inference inputs: image + metadata
- Strict-contract compatibility: compatible / 확인 필요

## Goal & Contribution
- Proposes FusionNetX as a multimodal framework for ISIC 2024 skin cancer detection.
- Combines CNN and Transformer image representations with metadata processed by tree-based classifiers.
- Emphasizes robustness under extreme class imbalance and patient-group generalization.

## Imbalance Handling
- Uses advanced sampling techniques according to the journal abstract.
- Uses stratified group cross-validation, which is important for patient-aware evaluation.
- Exact sampling and fold construction should be checked in the PDF before paper-level claims.

## Models
- Tabular: tree-based classifiers over metadata.
- Image: CNN and Transformer-based feature extractors.
- Fusion: image features integrated with metadata/tree model outputs; exact late/stacked fusion details 확인 필요.

## Metrics & Results
- Metrics: pAUC, private test score.
- Best reported result: cross-validation pAUC 0.18380 and private score 0.17295.
- Validation/test protocol: stratified group CV and ISIC 2024 private test score.
- Threshold selection: not specified in abstract; 확인 필요.

## Limitations
- Full text details are needed to verify fold grouping, preprocessing fit scope, and exact fusion mechanism.
- Private leaderboard score is useful but should not replace an internally reproducible patient-level protocol.
- Metadata feature list must be checked for diagnosis-derived or target-derived fields.

## Relevance to Our Study
- Very close reference for image + ordinary metadata multimodal baseline.
- Good comparison point for CNN/Transformer + metadata/tree ensemble design.
- Its reported patient-group CV makes it especially relevant if verified.

## Verification Notes
- Source: journal landing page and DOI page.
- Confirm whether the paper explicitly cites SLICE-3D and/or the ISIC 2024 triage paper.
