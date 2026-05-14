# ISIC 2024 Train-Only Multimodal Literature Review

조사일: 2026-05-04  
주제: ISIC 2024 Kaggle train dataset만 사용한 skin cancer detection 멀티모달 모델 연구를 위한 선행논문 조사  
핵심 조사 대상: dataset 불균형 극복 방법, image-tabular multimodal fusion 방법

---

## 1. 연구 배경

ISIC 2024 Kaggle Challenge는 3D Total Body Photography(3D-TBP)에서 추출한 피부 병변 crop image와 환자/병변 metadata를 함께 제공하는 binary classification 문제이다. 공식 train dataset은 SLICE-3D dataset이며, 총 401,059개 lesion tile로 구성된다.

가장 중요한 특징은 극단적인 class imbalance이다.

| 항목 | 개수 | 비율 |
|---|---:|---:|
| Benign / target 0 | 400,666 | 99.902% |
| Malignant / target 1 | 393 | 0.098% |
| Total | 401,059 | 100% |

따라서 accuracy 중심 평가는 부적절하며, malignant를 놓치지 않는 high-sensitivity 영역의 성능 평가가 중요하다. Kaggle 공식 metric도 `pAUC > 80% TPR`를 사용했다.

ISIC 2024 train dataset의 modality는 크게 두 가지이다.

- Image: 15mm x 15mm field-of-view lesion crop image
- Tabular metadata: age, sex, anatomical site, lesion size/color/shape 관련 WB360 measurements, patient_id, lesion_id, attribution 등

---

## 2. 논문 분석 요약

기존 단일 요약표는 논문 역할, 데이터셋, 모델 구조, 평가 결과가 한 번에 섞여 있어 가독성이 낮음. 아래 표들은 비교 목적별로 분할한 요약임. 긴 설명은 3장의 논문별 상세 분석 파일 참고.

### 2.1 한눈에 보는 논문 역할 요약

| 번호 | 논문/자료 | 우리 연구에서의 역할 | 핵심 키워드 | 상세 분석 |
|---|---|---|---|---|
| 1 | SLICE-3D Dataset, Scientific Data 2024 | ISIC 2024 train dataset의 1차 근거 | 3D-TBP, ultra-rare malignant, weak label | [3.1](literature_review_papers/3_1_slice_3d_dataset.md) |
| 2 | ISIC 2024 Automated Triage, npj Digital Medicine 2025 | Kaggle challenge 결과와 ablation reference | pAUC, WB360, patient-context, GBT late fusion | [3.2](literature_review_papers/3_2_automated_triage_3d_tbp.md) |
| 3 | Wang et al., Scientific Reports 2025 | 3D-TBP image + clinical feature late fusion 및 XAI 근거 | XGBoost, SHAP, CAM, nomogram | [3.3](literature_review_papers/3_3_wang_explainable_multimodal_ai.md) |
| 4 | MetaBlock, JBHI 2021 | metadata modulation baseline 근거 | scale, shift, gating | [3.4](literature_review_papers/3_4_metablock_metadata_modulation.md) |
| 5 | MMF-Net, Frontiers in Surgery 2022 | attention-based fusion baseline 근거 | self-attention, cross-attention | [3.5](literature_review_papers/3_5_mmf_net_cross_attention_fusion.md) |
| 6 | Yap et al., Experimental Dermatology 2018 | 초기 multimodal skin lesion classification 근거 | dermoscopy, clinical image, metadata | [3.6](literature_review_papers/3_6_yap_multimodal_skin_lesion_classification.md) |
| ~~7~~ | ~~Islam et al., Scientific Reports 2026~~ | patient metadata fusion 및 triage 근거 | patient-separated split, voting, specificity | [3.7](literature_review_papers/3_7_islam_patient_metadata_fusion.md) |
| ~~8~~ | ~~Nguyen et al., Sensors 2022~~ | imbalance-aware image + metadata baseline 근거 | class weight, soft-attention, metadata branch | [3.8](literature_review_papers/3_8_nguyen_soft_attention_imbalance.md) |
| ~~9~~ | ~~Focal Loss 2017 / Class-Balanced Loss 2019~~ | loss ablation 근거 | focal loss, effective number | [4.2](#42-loss-기반) |
| ~~10~~ | ~~GAN/Diffusion augmentation 관련 연구~~ | minority augmentation 후보 | synthetic positive, train-only generation | [4.4](#44-synthetic-augmentation) |

### 2.2 Dataset 및 imbalance 비교

| 번호 | 논문/자료 | Dataset / task | Imbalance 및 protocol 포인트 |
|---|---|---|---|
| 1 | SLICE-3D Dataset | ISIC 2024 train, 401,059 lesion tiles, binary malignant/benign | malignant 393개, 약 0.098%. 모델 학습 없음 |
| 2 | ISIC 2024 Automated Triage | ISIC 2024 Challenge set, binary malignant/benign | class 축소 없음. public/private leaderboard 및 pAUC > 80% TPR 중심 |
| 3 | Wang et al. 2025 | ISIC 2024 subset, 1,075 patients, 6-class lesion risk prediction | non-nevus class targeted augmentation. ISIC binary benchmark로 별도 비교 |
| 4 | MetaBlock | PAD-UFES-20, ISIC 2019 | weighted cross-entropy 및 stratified 5-fold CV |
| 5 | MMF-Net | PAD-UFES-20, 6-class smartphone clinical image + metadata | stratified 5-fold CV 및 on-the-fly augmentation |
| 6 | Yap et al. 2018 | 2,917 cases, dermoscopy + macroscopic image + metadata | binary melanoma와 5-class task 별도 평가 |
| 7 | Islam et al. 2026 | 39,623 lesions, 19,295 patients, DER/DSLR image + metadata | patient-separated split, augmentation, decision-level voting |
| 8 | Nguyen et al. 2022 | HAM10000, 10,015 images, 7 classes | augmentation to 53,573 images 및 class-weighted loss |
| 9 | Focal/Class-Balanced Loss | 특정 dataset 고정 없음 | long-tail classification 일반 loss |
| 10 | GAN/Diffusion augmentation | 연구별 skin lesion dataset | minority synthetic image 생성. train-only 조건 확인 필요 |

### 2.3 Model 및 fusion 방식 비교

| 번호 | 논문/자료 | Image branch | Tabular / metadata branch | Fusion 또는 학습 방식 |
|---|---|---|---|---|
| 2 | ISIC 2024 Automated Triage | EVA 2개 + EdgeNeXt ensemble | GBT, WB360, engineered feature, patient-context | NN output vector + metadata를 3개 GBT에 입력 |
| 3 | Wang et al. 2025 | HAM10000 transfer learning CNN | clinical feature + XGBoost | CNN probability vector + clinical feature late fusion |
| 4 | MetaBlock | CNN backbone의 last feature map | metadata modifier 생성 | feature map scale/shift/gating |
| 5 | MMF-Net | ResNet-50 | MLP metadata encoder | intra-modality self-attention + bidirectional cross-attention |
| 6 | Yap et al. 2018 | dermoscopic/clinical image CNN | patient metadata | multimodal classifier |
| 7 | Islam et al. 2026 | EfficientNet-B2 계열 | 7개 C4C risk factor + C4C risk score | feature concat 후 decision-level majority voting |
| 8 | Nguyen et al. 2022 | InceptionResNetV2, MobileNetV3Large + soft-attention | age, gender, localization dense branch | soft-attention output + metadata feature concat |
| 9 | Focal/Class-Balanced Loss | 모든 CNN/ViT에 적용 가능 | 해당 없음 | BCE 대체 loss 후보 |
| 10 | GAN/Diffusion augmentation | CNN classifier와 결합 가능 | 보통 없음 | train positive 기반 synthetic augmentation 후보 |

### 2.4 평가 지표 및 대표 결과 비교

| 번호 | 논문/자료 | 주요 metric | 대표 결과 | 해석 |
|---|---|---|---|---|
| 2 | ISIC 2024 Automated Triage | pAUC > 80% TPR, AUC, NNT | pAUC 0.1726/0.2, AUC 0.9668 | private leaderboard 기준 강한 challenge reference |
| 3 | Wang et al. 2025 | AUC, recall/F1, pFPR | multimodal AUC > 0.95, pFPR 0.17343 | late fusion + XAI 가능성 제시 |
| 4 | MetaBlock | BACC, ACC, AUC | 10개 실험 중 6개에서 최고 BACC | metadata modulation baseline 근거 |
| 5 | MMF-Net | BACC, aggregated AUC | BACC 0.775±0.022, AUC 0.947±0.007 | cross-attention fusion 근거 |
| 6 | Yap et al. 2018 | binary AUC, multiclass mAP | AUC 0.866 vs 0.784, mAP 0.729 vs 0.598 | early multimodal superiority 근거 |
| 7 | Islam et al. 2026 | sensitivity, specificity | voting SEN 99.50±1.18%, SPC 82.72±1.64% | high-sensitivity triage에서 specificity 개선 |
| 8 | Nguyen et al. 2022 | AUC, recall/F1, accuracy | abstract 기준 AUC 0.99, F1 0.81 | imbalance-aware loss + attention 근거 |
| 9 | Focal/Class-Balanced Loss | task별 metric | long-tailed dataset 성능 개선 보고 | ISIC 2024 loss ablation 후보 |
| 10 | GAN/Diffusion augmentation | accuracy, AUC 등 | dataset별 개선 보고 | 보조 실험 또는 ablation 후보 |

### 2.5 우리 연구 적용 시 주의점

| 항목 | 적용 가능성 | 주의점 |
|---|---|---|
| pAUC > 80% TPR | primary metric으로 직접 적용 가능 | 구현 정의, fold-wise reporting, AUC/F1/recall 병행 필요 |
| WB360 appearance metadata | tabular baseline 및 fusion 실험에 중요 | tile-only보다 항상 우수하다는 일반화 금지 |
| patient-context feature | ugly duckling feature 실험 가능 | patient-level split 및 fold-local 계산 필수 |
| image-only baseline | strict single-lesion setting의 핵심 비교군 | external dermoscopy data 사용 여부 분리 필요 |
| late fusion | ISIC 2024 winning solution과 가장 가까운 baseline | image score 생성 과정의 OOF/fold protocol 명시 필요 |
| metadata modulation / cross-attention | proposed multimodal model 후보 | late fusion 대비 ablation 필요 |
| imbalance-aware loss | image branch ablation 후보 | class weight는 train fold에서만 계산 필요 |
| synthetic augmentation | minority positive 보조 실험 후보 | train positive만 사용, artifact 및 overfitting 점검 필요 |

---

## 3. 주요 논문별 상세 분석

논문별 상세 분석은 문서 길이를 줄이고 읽기 속도를 개선하기 위해 별도 Markdown 파일로 분리했다. 이 장은 전체 흐름을 빠르게 파악하기 위한 링크 인덱스 역할을 한다.

| 번호 | 논문명 | 핵심 역할 | 상세 분석 링크 |
|---|---|---|---|
| 3.1 | SLICE-3D Dataset | ISIC 2024 train dataset의 공식 데이터셋 근거이며, ultra-rare malignant target, weak benign label, patient-level split 필요성을 정당화하는 1차 자료 | [상세 분석](literature_review_papers/3_1_slice_3d_dataset.md) |
| 3.2 | Automated Triage with 3D-TBP | ISIC 2024 challenge metric, metadata/patient-context ablation, image score와 tabular feature late fusion을 직접 뒷받침하는 핵심 baseline reference | [상세 분석](literature_review_papers/3_2_automated_triage_3d_tbp.md) |
| 3.3 | Wang et al. Explainable Multimodal AI | 3D-TBP image-derived prediction vector와 clinical feature를 XGBoost로 결합하는 late fusion 및 SHAP/CAM 기반 설명 가능성 설계의 근거이다. | [상세 분석](literature_review_papers/3_3_wang_explainable_multimodal_ai.md) |
| 3.4 | MetaBlock | metadata가 image feature map을 modulation하는 중간 fusion 구조의 대표 논문으로, 단순 concatenation을 넘어선 fusion baseline 후보이다. | [상세 분석](literature_review_papers/3_4_metablock_metadata_modulation.md) |
| 3.5 | MMF-Net | image branch와 metadata branch 사이의 self-attention 및 cross-attention fusion을 비교 대상으로 설계할 때 사용할 수 있는 선행 연구이다. | [상세 분석](literature_review_papers/3_5_mmf_net_cross_attention_fusion.md) |
| 3.6 | Yap et al. Multimodal Classification | dermoscopy, clinical image, patient metadata 결합이 image-only보다 나은 성능을 보인 초기 multimodal skin lesion classification 근거이다. | [상세 분석](literature_review_papers/3_6_yap_multimodal_skin_lesion_classification.md) |
| 3.7 | Islam et al. Patient Metadata Fusion | patient-separated split, metadata fusion, decision-level voting으로 high-sensitivity triage 성능을 개선한 최근 multimodal triage reference이다. | [상세 분석](literature_review_papers/3_7_islam_patient_metadata_fusion.md) |
| 3.8 | Nguyen et al. Soft Attention + Imbalance | soft attention, metadata branch, class-weighted loss를 함께 사용한 imbalance-aware image-tabular baseline 및 ablation 근거이다. | [상세 분석](literature_review_papers/3_8_nguyen_soft_attention_imbalance.md) |
