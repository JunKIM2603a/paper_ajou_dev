# 문헌 검색 로그

검색일: 2026-04-27

## 범위

- 기간: 2023-04-27 이후 발표 논문 또는 preprint(사전공개 논문)
- 대상: ISIC 2024 / SLICE-3D / 3D-TBP 기반 multimodal skin lesion / skin cancer 연구
- Seed 논문:
  - `The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection`
  - `Automated triage of cancer-suspicious skin lesions with 3D total-body photography`

## 검색 쿼리

| 날짜 | Query | 출처 | 메모 |
|---|---|---|---|
| 2026-04-27 | `"ISIC 2024" "multimodal" "tabular" "image" skin cancer arXiv` | web search | arXiv 및 publisher candidate(출판사 후보) |
| 2026-04-27 | `"SLICE-3D" "image" "tabular" "fusion" skin cancer` | web search | SLICE-3D direct candidate(직접 후보) |
| 2026-04-27 | `"The SLICE-3D dataset" "cited by" "multimodal"` | web search | citation candidate(인용 후보), ResearchGate snippet(발췌) |
| 2026-04-27 | `"Automated triage of cancer-suspicious" "cited by"` | web search | triage paper citation candidate(선별 논문 인용 후보) |
| 2026-04-27 | `"FusionNetX" "partial" "0.18380"` | web search | journal article 발견 |
| 2026-04-27 | `"Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data"` | web search | Scientific Reports article 발견 |

## 후보 논문

| Paper | 출처 | Seed 인용 관계 | 포함 여부 | 이유 |
|---|---|---|---|---|
| Automated triage of cancer-suspicious skin lesions with 3D total-body photography | Nature / npj Digital Medicine | SLICE-3D를 인용; seed paper 2 | yes | ISIC 2024 official outcome(공식 결과)/ablation(구성요소 제거 비교) reference |
| FusionNetX: A highly effective multimodal framework for skin cancer detection | Journal of Computer Science and Cybernetics | SLICE-3D / ISIC 2024 citation 확인 필요 | yes | 직접적인 ISIC 2024 image+metadata multimodal paper(영상+메타데이터 멀티모달 논문) |
| Multimodal system for skin cancer detection | arXiv 2601.14822 | reference를 통한 SLICE-3D 인용 확인 필요 | yes | pAUC를 보고하는 직접적인 image+metadata multimodal system(멀티모달 시스템) |
| Hybrid Ensemble of Segmentation-Assisted Classification and GBDT... | arXiv 2506.03420 | 직접 SLICE-3D 사용; 명시적 citation 확인 필요 | yes | hybrid image+metadata ensemble(하이브리드 영상+메타데이터 앙상블) 및 synthetic lesion augmentation(합성 병변 증강) |
| Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data | Scientific Reports | SLICE-3D / triage context(선별 맥락) 인용 가능성 있음; 정확한 reference 확인 필요 | yes | peer-reviewed(동료심사) explainable image+clinical fusion(설명가능 영상+임상 융합) |
| Dual-stage segmentation and classification framework for skin lesion analysis using deep neural network | Digital Health / SAGE | SLICE-3D 사용; 정확한 seed citation 확인 필요 | yes | SLICE tabular-only(표형 단독) vs fusion(융합) pAUC contrast(대조) 포함 |
| A Personalized Multimodal Federated Learning Framework for Skin Cancer Diagnosis | MDPI Electronics | ISIC 2024 challenge citation; seed citation 확인 필요 | yes, related | federated/missing-modality setting(연합학습/누락 모달리티 설정)이며 직접적인 strict baseline은 아님 |
| A Novel Transfer Learning Approach for Skin Cancer Classification on ISIC 2024 3D Total Body Photographs | International Journal of Imaging Systems and Technology | ISIC 2024 / SLICE-3D citation 확인 필요 | no | image-only(영상 단독)이거나 multimodal 여부가 불명확함; 약한 related work로만 유용 |
| Improved Skin Cancer Detection with 3D Total Body Photography: Integrating AI Algorithms for Precise Diagnosis | Research Square | ISIC 2024 / 3D-TBP; seed citation 확인 필요 | low-confidence candidate(낮은 신뢰도 후보) | preprint detail(세부 정보)이 부족함; verification queue(검증 대기열)에만 유지 |
| Skin region images extracted from 3D total body photographs for lesion detection / iToBoS dataset | Scientific Data / arXiv | related 3D-TBP dataset; SLICE-3D와 대비됨 | no | dataset descriptor(데이터셋 설명 논문)이며 multimodal classification model은 아님 |
| HCHS-Net: A Multimodal Handcrafted Feature and Metadata Framework... | MDPI Biomimetics | ISIC 2024 아님; PAD-UFES-20 | no | 일반 multimodal metadata paper로는 유용하지만 현재 dataset scope(데이터셋 범위) 밖 |
| SkinEHDLF hybrid deep learning approach | Scientific Reports | ISIC 2024 언급 | no | image-centric(영상 중심)으로 보이며 dataset description의 balance claim(균형 주장)에 주의 필요 |

## 검증 대기열

| 항목 | 상태 | 메모 |
|---|---|---|
| 각 candidate(후보)의 Google Scholar cited-by relation(인용 관계) | 확인 필요 | Web snippet은 직접 dataset 사용을 식별하지만, paper text에 사용할 경우 formal cited-by(공식 인용 관계)를 수동 확인해야 한다 |
| FusionNetX의 정확한 patient-level split(환자 단위 분할) | partial(부분 확인) | abstract는 stratified group CV(층화 그룹 교차검증)를 언급함; 상세 split source(분할 출처)는 PDF에서 확인해야 한다 |
| arXiv 2601.14822의 정확한 split/threshold source(분할/임계값 출처) | partial | arXiv abstract와 review/source snippet은 metric(지표)을 보고함; paper claim 전 PDF detail 확인 필요 |
| arXiv 2506.03420의 diagnosis-informed relabeling leakage risk(진단 정보 기반 재라벨링 누수 위험) | 확인 필요 | diagnosis-informed external-data harmonization(진단 정보 기반 외부 데이터 조화)을 사용하므로 strict baseline과 분리해야 한다 |
| Scientific Reports 2025 XAI paper의 pathology-derived feature risk(병리 유래 특징 위험) | major risk(주요 위험) | 보고 feature list에 `mel_thick_mm`가 포함됨; 제거 및 재감사 전에는 strict-compatible(엄격 호환)로 취급하지 않는다 |
| Scientific Reports 2025 XAI paper의 pFPR wording | 확인 필요 | 논문은 pFPR 0.17343을 보고함; 비교 전 pAUC@TPR>=0.80과 신중히 mapping해야 한다 |
| PMM-FL ISIC 2024 lesion count typo/risk(병변 수 오타/위험) | 확인 필요 | source text 한 곳에서 401,059가 아니라 40,159로 보고됨; correction(정정) 없이 dataset fact(데이터셋 사실)로 사용하지 않는다 |
