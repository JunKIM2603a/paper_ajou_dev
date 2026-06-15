# Dermatology AI Literature Review for ISIC 2024 Multimodal Research

조사일: 2026-05-18  
주제: ISIC 2024 3D-TBP skin cancer detection 및 image-tabular multimodal 연구 확장을 위한 최신 dermatology AI 논문 정리  
핵심 조사 대상: [The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection](https://www.nature.com/articles/s41597-024-03743-w.pdf)을 인용한 논문 중 인용 횟수 및 신뢰성 있는 논문
keyword: foundation model, 3D-TBP dataset, multimodal image pair/metadata, imbalance-aware model, dataset transparency

---

## 1. 연구 배경

ISIC 2024 Kaggle Challenge는 3D Total Body Photography(3D-TBP)에서 추출한 피부 병변 crop image와 환자/병변 metadata를 함께 제공하는 binary malignant/benign classification 문제이다. 공식 train dataset은 SLICE-3D dataset이며, 총 401,059개 lesion tile로 구성된다.

| 항목 | 개수 | 비율 |
|---|---|---|
| Benign / target 0 | 400,666 | 99.902% |
| Malignant / target 1 | 393 | 0.098% |
| Total | 401,059 | 100% |

따라서 단순 accuracy 중심 평가는 부적절하며, malignant를 놓치지 않는 high-sensitivity 영역에서 specificity와 biopsy burden을 함께 보는 평가가 필요하다. Kaggle 공식 metric도 `pAUC > 80% TPR`를 사용했다.

5월 14일 문헌 정리가 image-tabular fusion architecture와 train-only protocol의 직접 근거를 정리했다면, 이번 5월 28일 문헌 묶음은 더 넓은 연구 배경을 보완한다. 특히 다음 네 가지 방향에서 ISIC 2024 연구의 introduction, related work, discussion을 강화할 수 있다.

- Dermatology foundation model: PanDerm처럼 TBP tile, clinical, dermoscopy, dermatopathology를 하나의 representation space로 학습하는 방향
- 3D-TBP 및 longitudinal dataset: 단일 lesion crop보다 patient context, lesion-to-lesion comparison, time-series 변화가 중요하다는 근거
- Multimodal imaging dataset: clinical close-up + dermoscopy pair, metadata, hierarchical diagnosis benchmark의 필요성
- Dataset transparency: class imbalance, label quality, skin tone documentation, hidden proxy risk를 명시하는 dataset documentation 관점

---

## 2. 논문 분석 요약

### 2.1 한눈에 보는 논문 역할 요약

| 번호 | 논문/자료 | 우리 연구에서의 역할 | 핵심 키워드 | 상세 분석 |
|---:|---|---|---|---|
| 1 | PanDerm / A multimodal vision foundation model for clinical dermatology | 최신 dermatology foundation model 및 multimodal pretraining 근거 | foundation model, 3D-TBP, multimodal, label-efficient | [3.1](literature_review_papers/3_1_panderm_foundation_model.md) |
| 2 | SkinEHDLF | ISIC 2024 기반 hybrid image ensemble 및 class-weighted loss 참고 | ConvNeXt, EfficientNetV2, Swin, attention fusion | [3.2](literature_review_papers/3_2_skinehdlf_hybrid_deep_learning.md) |
| 3 | DERM12345 | 세분화된 dermoscopy subclass dataset 및 dataset 확장 근거 | 40 subclasses, dermoscopy, multiclass | [3.3](literature_review_papers/3_3_derm12345_dataset.md) |
| 4 | MILK10k | clinical close-up + dermoscopy image-pair benchmark 근거 | multimodal imaging, hierarchical diagnosis, Siamese | [3.4](literature_review_papers/3_4_milk10k_multimodal_imaging.md) |
| 5 | Skin Region Images from 3D-TBP | 병변 crop 외 주변 피부 context와 lesion detection/localisation 근거 | 3D-TBP, skin-region tile, detection | [3.5](literature_review_papers/3_5_skin_region_3d_tbp_dataset.md) |
| 6 | Optimized Five-Stream CNN | multi-stream image feature 및 GAN balance 참고, 성능 해석 주의 | multi-stream CNN, GAN augmentation, dermoscopy | [3.6](literature_review_papers/3_6_optimized_five_stream_cnn.md) |
| 7 | Dataset Nutrition Label | ISIC 2024/SLICE-3D의 dataset transparency 및 risk documentation 근거 | bias, transparency, skin tone, label risk | [3.7](literature_review_papers/3_7_dataset_nutrition_label.md) |
| 8 | OHSU MoleMapper Release | user-captured smartphone lesion image dataset 및 self-supervised pretraining 후보 | smartphone, consumer image, metadata | [3.8](literature_review_papers/3_8_ohsu_molemapper_release.md) |
| 9 | Longitudinal Tile + Dermoscopy Dataset | patient context, longitudinal change, tile-dermoscopy 연결 근거 | longitudinal, metadata, dermoscopy, 3D-TBP | [3.9](literature_review_papers/3_9_longitudinal_tile_dermoscopy_dataset.md) |
| 10 | IEEE Access Segmentation/Classification Review | segmentation/classification challenge와 future direction 정리 | review, segmentation, classification, deployment | [3.10](literature_review_papers/3_10_skin_cancer_segmentation_classification_review.md) |
| 11 | AI Dermatology Frontiers Review | skin cancer detection AI 전반의 넓은 관련 연구 배경 | review, clinical integration, XAI, transformer | [3.11](literature_review_papers/3_11_ai_dermatology_frontiers_review.md) |
| 12 | AI in Non-Invasive Detection of Melanoma | melanoma triage, imaging modality, clinical validation 배경 | melanoma, non-invasive, clinical image, dermoscopy | [3.12](literature_review_papers/3_12_non_invasive_melanoma_ai_review.md) |
| 13 | FoMoSkinNet | non-dermoscopic ISIC 2024 subset 기반 dual-stream baseline 참고 | focal modulation, local feature, balanced subset | [3.13](literature_review_papers/3_13_fomoskinnet_dual_stream.md) |
| 14 | AcuSim | 직접 관련도 낮은 synthetic RGB-D localization 자료, 보조 참고 | synthetic dataset, RGB-D, domain randomization | [3.14](literature_review_papers/3_14_acusim_synthetic_dataset.md) |
| 15 | Medical Video Generation | longitudinal progression synthetic data의 보조 참고 | diffusion, progression simulation, prompt control | [3.15](literature_review_papers/3_15_medical_video_generation.md) |

### 2.2 Dataset 및 imbalance 비교

| 번호 | 논문/자료 | Dataset / task | Imbalance 및 protocol 포인트 |
|---:|---|---|---|
| 1 | PanDerm | 11개 기관, 4개 modality, 2,149,706개 unlabeled image pretraining; 28개 downstream dataset | self-supervised pretraining으로 희소 label 문제 완화. TBP screening에서 216 malignant vs 197,716 benign 처리 |
| 2 | SkinEHDLF | ISIC 2024 3D-TBP image 기반 binary/multi-class classification | class-weighted loss와 augmentation 사용. reported performance는 외부 검증 관점에서 보수적으로 인용 필요 |
| 3 | DERM12345 | 12,345 dermoscopy image, 40 subclass | patient-level 80/20 split과 class balancing 고려. baseline accuracy가 낮아 subclass difficulty 근거로 유용 |
| 4 | MILK10k | 5,240 lesions, clinical close-up + dermoscopy 10,480 image pair, 48 diagnosis | 5-fold stratified split, inverse class-frequency weighting, label smoothing |
| 5 | Skin Region 3D-TBP | 100명, 16,954개 skin-region tile, suspicious lesion bounding box | dataset descriptor. benign predominance와 annotation inconsistency 가능성 언급 |
| 6 | Five-Stream CNN | HAM10000, ISIC 2024, ISIC 2017, ISIC 2016 | GAN 기반 class balance 및 전처리. 매우 높은 accuracy 수치는 leakage/검증 설계 확인 필요 |
| 7 | Dataset Nutrition Label | SLICE-3D/ISIC 2024 case study | imbalance를 보정하지 않고 risk category로 문서화 |
| 8 | MoleMapper | 4,158명, 27,499 mole crop, 7,305 nearby skin patch, 1,000 zone image | unlabeled consumer-captured dataset. self-supervised pretraining 및 품질/편향 분석 후보 |
| 9 | Longitudinal Tile + Dermoscopy | 480명, 250,162 tile image, 9,389 dermoscopy image, 2-7회 follow-up 포함 | model-level imbalance 처리는 없지만 cohort/metadata 기반 subgroup 분석 가능 |
| 10 | Segmentation/Classification Review | PH2, ISIC 2016-2024, HAM10000, Derm7pt, Fitzpatrick17k 등 | class imbalance와 dataset diversity를 주요 challenge로 정리 |
| 11 | AI Dermatology Review | 공개 dataset 및 임상 적용 사례 review | data diversity, augmentation, bias, clinical integration 한계 논의 |
| 12 | Non-Invasive Melanoma Review | clinical, dermoscopy, RCM/OCT 등 문헌 dataset | skin type distribution, malignant ratio, external validation 부족을 한계로 제시 |
| 13 | FoMoSkinNet | ISIC 2024 subset, benign 15,000 + malignant 15,000 균형 dataset | 원래 ISIC 2024의 극단적 imbalance를 balanced subset으로 완화. sampling validity 확인 필요 |
| 14 | AcuSim | 504 synthetic anatomical model, 63,936 RGB-D image | 피부암 dataset이 아니며 synthetic diversity/domain randomization 중심 |
| 15 | Medical Video Generation | CheXpert, MIMIC-CXR, DR, ISIC 2024, ISIC 2018 | class imbalance보다 longitudinal progression data 부족을 다룸 |

### 2.3 Model 및 fusion 방식 비교

| 번호 | 논문/자료 | Image branch | Tabular / metadata branch | Fusion 또는 학습 방식 |
|---:|---|---|---|---|
| 1 | PanDerm | ViT-Large visual encoder, masked latent alignment, CLIP-Large teacher | 일부 TBP screening에서 lesion measurement/metadata 사용 | 여러 dermatology modality를 공통 embedding으로 통합, 일부 task에서 image feature + metadata 결합 |
| 2 | SkinEHDLF | ConvNeXt, EfficientNetV2, Swin Transformer | 명확한 tabular input branch 없음 | 세 backbone feature를 adaptive attention-based fusion으로 결합 |
| 3 | DERM12345 | ResNet50, Xception, InceptionResNetV2 baseline | CSV metadata는 label/taxonomy 중심 | 단일 dermoscopy image classification dataset |
| 4 | MILK10k | ResNet50 Siamese image encoder for close-up and dermoscopy | age/sex/skin tone/site metadata 제공, baseline은 image pair 중심 | 두 image branch feature concat 후 projection/classifier |
| 5 | Skin Region 3D-TBP | 새 모델 없음, detection/localisation annotation 제공 | anatomical location, demographics, sun damage score 등 metadata | image tile + YOLO/COCO annotation + metadata의 dataset-level 결합 |
| 6 | Five-Stream CNN | color, edge, texture, local frequency, gradient stream 기반 OFSCNN | 없음 | 다중 feature stream을 CNN 내부에서 결합 |
| 7 | Dataset Nutrition Label | 해당 없음 | dataset metadata/limitation/risk 구조화 | 모델 fusion이 아니라 dataset documentation framework |
| 8 | MoleMapper | 모델 없음 | participant metadata 제공 | mole crop + nearby skin patch + contextual zone image + metadata 제공 |
| 9 | Longitudinal Dataset | VECTRA WB360 tile, corresponding dermoscopy | demographics, history, sun exposure, naevus count, site metadata | tile + dermoscopy + longitudinal time point + participant metadata 연결 |
| 10 | Segmentation/Classification Review | CNN, U-Net, transformer, hybrid model review | 주요 대상 아님 | pretrained CNN fusion, hybrid CNN-transformer, multimodal 접근 논의 |
| 11 | AI Dermatology Review | CNN, transformer, segmentation, XAI model review | 독립 tabular branch 없음 | multimodal integration과 clinical workflow 통합을 review 수준에서 논의 |
| 12 | Non-Invasive Melanoma Review | clinical, dermoscopy, RCM/OCT AI model review | 일부 문헌에서 metadata 활용 언급 | imaging modality 및 metadata 조합 연구 소개 |
| 13 | FoMoSkinNet | FocalNet global branch + LFNet local branch | 없음 | global/local feature를 hybrid feature fusion으로 결합 |
| 14 | AcuSim | VGG19 feature extractor 기반 multitask CNN | coordinate/annotation metadata | RGB-D image, depth, coordinate metadata를 localization pipeline에서 사용 |
| 15 | Medical Video Generation | Stable Diffusion PIE + SEINE transition generation | clinical report text prompt | text prompt + region mask + initial image + diffusion intermediate state 결합 |

### 2.4 평가 지표 및 대표 결과 비교

| 번호 | 논문/자료 | 주요 metric | 대표 결과 | 해석 |
|---:|---|---|---|---|
| 1 | PanDerm | AUROC, AUPR, weighted F1, balanced accuracy, sensitivity, reader study | TBP screening sensitivity 0.893, 불필요한 dermoscopy 약 60.8% 감소 | ISIC 2024 discussion에서 foundation model과 triage burden reduction 근거 |
| 2 | SkinEHDLF | accuracy, precision, recall, F1, AUROC, specificity, MCC | binary AUROC 99.8%, accuracy 98.76% | 강한 성능 주장이나 external validation 및 split protocol 확인 필요 |
| 3 | DERM12345 | weighted accuracy, technical validation | baseline weighted accuracy 0.50-0.59 | fine-grained diagnosis difficulty와 dataset 확장 필요성 근거 |
| 4 | MILK10k | recall, specificity, top-1/top-3 accuracy, hierarchical distance | 48-class top-1 53.6%, top-3 67.7% | 계층형 진단과 multimodal image pair task의 난도 제시 |
| 5 | Skin Region 3D-TBP | annotation quality, dermatologist review | classifier metric 없음 | lesion detection/localisation dataset 근거 |
| 6 | Five-Stream CNN | accuracy, precision, recall, F1, 5-fold CV | ISIC 2024 accuracy 99.9% 보고 | 높은 성능은 보조 참고로만 사용하고 직접 baseline 근거로 과신 금지 |
| 7 | Dataset Nutrition Label | risk category, SME review | SLICE-3D risk와 적합/부적합 use case 정리 | dataset section 및 limitation 강화에 유용 |
| 8 | MoleMapper | curation, HIPAA/PHI filtering, access process | classification metric 없음 | consumer image pretraining/quality analysis 후보 |
| 9 | Longitudinal Dataset | technical validation, metadata consistency | 신규 진단 모델 metric 없음 | patient-context/longitudinal 연구 필요성 근거 |
| 10 | Segmentation/Classification Review | accuracy, Dice, IoU, sensitivity, specificity 문헌 비교 | 자체 benchmark 없음 | challenge/future direction 인용 자료 |
| 11 | AI Dermatology Review | accuracy, precision, recall, F-score 문헌 비교 | 자체 benchmark 없음 | broad related work 및 clinical integration 배경 |
| 12 | Non-Invasive Melanoma Review | sensitivity, specificity, PPV, NPV, AUC 문헌 비교 | 자체 benchmark 없음 | melanoma triage와 clinical validation discussion |
| 13 | FoMoSkinNet | accuracy, F1, specificity, precision, sensitivity, AUC | ISIC 2024 accuracy 98.85%, F1 98.84% | balanced subset 기반이라 원본 imbalance setting과 구분 필요 |
| 14 | AcuSim | classification accuracy, localization ratio, RMSE | validation accuracy 99.73%, 5mm 이내 92.86% | 피부암 직접 근거는 낮고 synthetic dataset 방법론 보조 참고 |
| 15 | Medical Video Generation | CLIP-I, confidence score, clinician preference, MAE | skin task confidence 0.453, CLIP-I 0.958 | longitudinal synthetic augmentation 아이디어의 보조 근거 |

### 2.5 우리 연구 적용 시 주의점

| 항목 | 적용 가능성 | 주의점 |
|---|---|---|
| Foundation model pretraining | PanDerm을 통해 ISIC 2024 image-only 또는 multimodal model의 장기 확장 방향 제시 가능 | train-only 실험과 external pretraining을 명확히 분리해야 함 |
| 3D-TBP tile context | SLICE-3D 외 3D-TBP region/longitudinal dataset을 통해 patient-context 필요성 강화 | dataset 간 촬영 범위, label definition, annotation granularity가 다름 |
| Image pair multimodality | MILK10k는 clinical close-up + dermoscopy pair fusion 근거로 활용 가능 | ISIC 2024 train에는 dermoscopy pair가 없으므로 직접 비교 baseline으로 쓰면 안 됨 |
| Metadata 및 patient context | longitudinal dataset과 nutrition label 모두 metadata의 중요성과 risk를 보여줌 | metadata leakage, hidden proxy, patient-level split을 함께 논의해야 함 |
| Imbalance-aware model | SkinEHDLF, FoMoSkinNet, Five-Stream CNN의 class weighting/balancing 참고 가능 | balanced subset 결과를 원본 0.098% malignant setting 성능처럼 해석하면 안 됨 |
| Reported high accuracy | 여러 2025-2026 모델 논문이 매우 높은 accuracy를 보고 | pAUC, sensitivity-specificity tradeoff, external validation 여부를 중심으로 재해석 필요 |
| Dataset transparency | Dataset Nutrition Label은 limitation section의 직접 근거 | skin tone documentation, weak label, rare subtype exclusion 등은 성능 외 핵심 위험 |
| Synthetic augmentation | Medical Video Generation, AcuSim은 synthetic data 방법론의 보조 참고 | 피부암 진단 직접 근거가 약하므로 주장의 중심에 두지 않음 |

---

## 3. 주요 논문별 상세 분석

논문별 상세 분석은 문서 길이를 줄이고 읽기 속도를 개선하기 위해 별도 Markdown 파일로 분리했다. 이 장은 전체 흐름을 빠르게 파악하기 위한 링크 인덱스 역할을 한다.

| 번호 | 논문명 | 핵심 역할 | 추가 논의 | 상세 분석 링크 |
|---:|---|---|---|---|
| 3.1 | A multimodal vision foundation model for clinical dermatology / PanDerm | dermatology foundation model과 3D-TBP screening의 최신 핵심 reference | Nature Medicine published PDF를 우선 인용하고 arXiv v1은 참고 preprint로 유지 | [상세 분석](literature_review_papers/3_1_panderm_foundation_model.md) |
| 3.2 | SkinEHDLF | ISIC 2024 기반 hybrid backbone ensemble 참고 | 성능 수치의 split/protocol 검증 필요 | [상세 분석](literature_review_papers/3_2_skinehdlf_hybrid_deep_learning.md) |
| 3.3 | DERM12345 | 40 subclass dermoscopy dataset으로 fine-grained classification difficulty 제시 | ISIC 2024 binary task와 taxonomy 차이 명시 | [상세 분석](literature_review_papers/3_3_derm12345_dataset.md) |
| 3.4 | MILK10k | clinical close-up + dermoscopy image-pair benchmark와 hierarchical diagnosis 근거 | ISIC 2024에는 dermoscopy pair가 없으므로 배경/확장 근거로 활용 | [상세 분석](literature_review_papers/3_4_milk10k_multimodal_imaging.md) |
| 3.5 | Skin Region Images from 3D-TBP | 주변 피부 context와 detection/localisation dataset 근거 | lesion crop-only 한계 논의에 유용 | [상세 분석](literature_review_papers/3_5_skin_region_3d_tbp_dataset.md) |
| 3.6 | Optimized Five-Stream CNN | multi-stream handcrafted-like image feature와 GAN balancing 참고 | extremely high accuracy 과신 금지 | [상세 분석](literature_review_papers/3_6_optimized_five_stream_cnn.md) |
| 3.7 | Dataset Nutrition Label | ISIC 2024/SLICE-3D의 bias, limitation, transparency 근거 | dataset documentation과 limitation section에 중요 | [상세 분석](literature_review_papers/3_7_dataset_nutrition_label.md) |
| 3.8 | OHSU MoleMapper | consumer-captured smartphone lesion image release | unlabeled data라 supervised benchmark로는 부적합 | [상세 분석](literature_review_papers/3_8_ohsu_molemapper_release.md) |
| 3.9 | Longitudinal Tile + Dermoscopy Dataset | tile, dermoscopy, metadata, time-series를 연결한 patient-context reference | ugly duckling/longitudinal feature 확장 논의에 유용 | [상세 분석](literature_review_papers/3_9_longitudinal_tile_dermoscopy_dataset.md) |
| 3.10 | Segmentation and Classification Review | skin cancer segmentation/classification challenge 종합 | review 논문이므로 직접 성능 baseline은 아님 | [상세 분석](literature_review_papers/3_10_skin_cancer_segmentation_classification_review.md) |
| 3.11 | AI Dermatology Frontiers Review | skin cancer detection AI 전반의 broad related work | clinical integration, XAI, data diversity 논의 | [상세 분석](literature_review_papers/3_11_ai_dermatology_frontiers_review.md) |
| 3.12 | AI in Non-Invasive Detection of Melanoma | melanoma triage와 imaging modality별 AI 배경 | clinical validation 부족과 dataset bias 논의 | [상세 분석](literature_review_papers/3_12_non_invasive_melanoma_ai_review.md) |
| 3.13 | FoMoSkinNet | non-dermoscopic ISIC 2024 subset dual-stream model reference | balanced subset setting을 원본 challenge와 분리 | [상세 분석](literature_review_papers/3_13_fomoskinnet_dual_stream.md) |
| 3.14 | AcuSim | synthetic RGB-D dataset 및 localization pipeline 보조 참고 | 피부암/피부과 진단 직접 관련도 낮음 | [상세 분석](literature_review_papers/3_14_acusim_synthetic_dataset.md) |
| 3.15 | Medical Video Generation | medical image progression simulation과 diffusion augmentation 보조 참고 | 피부 image는 검증 도메인 중 하나라 중심 reference로는 부적합 | [상세 분석](literature_review_papers/3_15_medical_video_generation.md) |
