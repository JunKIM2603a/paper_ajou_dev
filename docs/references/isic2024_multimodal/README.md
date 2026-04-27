# ISIC2024 Multimodal Literature Review

이 폴더는 ISIC2024 / SLICE-3D / 3D-TBP 기반 멀티모달 피부암 논문조사 결과를 축적하는 공간이다.

기본 관점은 `lesion image + ordinary inference-time tabular metadata -> malignant probability`이다. LUPI, privileged supervision, diagnosis text, pathology-derived text, `iddx_full` 기반 방법은 기본 baseline이 아니라 candidate 또는 related idea로 분리한다.

## Files

- `papers/`: 논문별 Markdown 요약
- `comparison_table.md`: 논문 간 비교표
- `search_log.md`: 검색 쿼리, 후보 논문, 제외 이유, 확인 필요 사항

## Summary Length

논문별 요약은 기본 45-55라인을 목표로 하고, 최대 70라인 미만으로 유지한다.

## Required Extraction Fields

- Citation
- Seed citation
- Dataset / Task / Modalities
- Inference inputs
- Strict-contract compatibility
- Goal & Contribution
- Imbalance Handling
- Tabular Model
- Image Model
- Fusion
- Metrics & Results
- Limitations
- Relevance to Our Study
- Verification Notes

## Strict-Contract Rule

`iddx_full`, diagnosis text, pathology-derived context, oracle diagnosis label을 inference input으로 요구하는 논문은 이 프로젝트의 strict multimodal baseline과 직접 비교하지 않는다. 이런 논문은 related work, limitation discussion, 또는 candidate method context로만 사용한다.
