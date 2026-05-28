# PDF 논문 요약: OpenAlex 인용 수와 출처 신뢰도 기준 우선순위

- 작성일: 2026-05-18
- 대상 PDF: `C:\Users\user\Desktop\00_Temp\20260518_paper` 폴더의 15개 PDF
- 정렬 기준: 출처/저널 신뢰도 우선, 같은 신뢰도 묶음 안에서는 OpenAlex `cited_by_count` 우선, 피부 AI 관련도는 활용 판단용으로 표시
- OpenAlex 조회일: 2026-05-18
- 주의: OpenAlex 인용 수는 조회 시점의 `cited_by_count`이며, 최신 논문은 실제 영향도 대비 낮게 잡힐 수 있음.

## 우선순위 요약표

| 순위 | 논문명 | 출처 | 출판연도 | DOI | OpenAlex 인용 수 | 우선순위 근거 | 주제 관련도 |
|---:|---|---|---|---|---:|---|---|
| 1 | A General-Purpose Multimodal Foundation Model for Dermatology / A multimodal vision foundation model for clinical dermatology | arXiv PDF; Nature Medicine published version | 2024 PDF; 2025 journal | 10.48550/arXiv.2410.15038; 10.1038/s41591-025-03747-y | 59 (journal version) | Nature Medicine published version이 확인되고, 직접적인 피부과 멀티모달 foundation model 논문 중 인용 수가 가장 높음. | 매우 높음 |
| 2 | SkinEHDLF a hybrid deep learning approach for accurate skin cancer classification in complex systems | Scientific Reports | 2025 | 10.1038/s41598-025-98205-7 | 24 | Nature Portfolio 계열 저널, 피부암 분류 모델 논문, 인용 수 높음. | 매우 높음 |
| 3 | DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses | Scientific Data | 2024 | 10.1038/s41597-024-04104-3 | 17 | Nature Portfolio 데이터 저널, 세분화된 피부 병변 데이터셋으로 인용 수 높음. | 매우 높음 |
| 4 | MILK10k: A Hierarchical Multimodal Imaging-Learning Toolkit for Diagnosing Pigmented and Nonpigmented Skin Cancer and its Simulators | Journal of Investigative Dermatology | 2026 issue; 2025 online | 10.1016/j.jid.2025.06.1594 | 4 | 피부과 전문 저널 JID, 멀티모달 이미지쌍과 계층 진단 벤치마크 제공. | 매우 높음 |
| 5 | Skin region images extracted from 3D total body photographs for lesion detection | Scientific Data | 2025 | 10.1038/s41597-025-05483-x | 2 | Nature Portfolio 데이터 저널, 3D total-body photography 기반 병변 탐지 데이터셋. | 높음 |
| 6 | Automatic melanoma detection using an optimized five-stream convolutional neural network | Scientific Reports | 2025 | 10.1038/s41598-025-05675-w | 1 | Nature Portfolio 계열 저널, 피부암 이미지 분류 모델 논문이나 매우 높은 성능 수치는 외부 검증 관점에서 주의 필요. | 매우 높음 |
| 7 | Improving dataset transparency in dermatologic Artificial Intelligence using a dataset nutrition label | npj Digital Medicine | 2025 | 10.1038/s41746-025-02125-9 | 1 | Nature Portfolio/npj 계열, 모델 성능 논문은 아니지만 데이터 편향 및 투명성 기준으로 중요. | 높음 |
| 8 | New Release of User-Captured Images from the OHSU Melanoma MoleMapper Project | Scientific Data | 2025 | 10.1038/s41597-025-05552-1 | 0 | Nature Portfolio 데이터 저널, 사용자 촬영 스마트폰 피부 병변 데이터셋. | 높음 |
| 9 | A longitudinal dataset of tile and corresponding dermoscopic images with metadata for identifying skin cancers | Scientific Data | 2025 | 10.1038/s41597-025-05880-2 | 0 | Nature Portfolio 데이터 저널, longitudinal tile/dermoscopy/metadata를 연결한 데이터셋. | 매우 높음 |
| 10 | Segmentation and Classification of Skin Cancer Diseases Based on Deep Learning: Challenges and Future Directions | IEEE Access | 2025 | 10.1109/ACCESS.2025.3569170 | 9 | peer-reviewed review 논문이며 인용 수는 높지만, 신규 모델/데이터셋이 아닌 종설. | 높음 |
| 11 | AI dermatology: Reviewing the frontiers of skin cancer detection technologies | Intelligent Oncology | 2025 | 10.1016/j.intonc.2025.03.002 | 7 | peer-reviewed review, 피부암 AI 전반을 넓게 다룸. | 높음 |
| 12 | Artificial Intelligence in the Non-Invasive Detection of Melanoma | Life | 2024 | 10.3390/life14121602 | 7 | MDPI Life review, 비침습 melanoma AI 문헌을 폭넓게 정리. | 높음 |
| 13 | FoMoSkinNet: A Dual-Stream Deep Learning Model for Early Detection and Classification of Non-Dermoscopic Skin Lesions via Focal Modulation and Local Feature Integration | IEEE Access | 2026 | 10.1109/ACCESS.2026.3674219 | 0 | 최신 peer-reviewed 모델 논문이나 출판 직후라 인용 수 없음. | 매우 높음 |
| 14 | AcuSim: A Synthetic Dataset for Cervicocranial Acupuncture Points Localisation | Scientific Data | 2025 | 10.1038/s41597-025-04934-9 | 6 | Nature Portfolio 데이터 저널이지만 피부암/피부과 AI와 직접 관련성이 낮아 활용 우선순위는 낮음. | 낮음 |
| 15 | Medical Video Generation for Disease Progression Simulation | arXiv | 2024 | 10.48550/arXiv.2411.11943 | 2 | preprint이며 피부 이미지는 검증 도메인 중 하나로만 포함. | 중간 |

## 1. A General-Purpose Multimodal Foundation Model for Dermatology

- 파일: `33499_A_General_Purpose_Multim.pdf`
- 출처/인용: PDF는 arXiv v1이며, 동일 연구의 published version은 Nature Medicine `A multimodal vision foundation model for clinical dermatology`로 확인됨; OpenAlex 인용 수는 journal version 59, arXiv version 1.

- 목표와 기여: PanDerm이라는 범용 멀티모달 피부과 foundation model을 제안해 피부암 선별, 진단, 분할, 병변 변화 추적, 예후 예측까지 하나의 표현 학습 모델로 지원.
- Dataset 정보: 11개 기관, 4개 modality(TBP tile, dermatopathology, clinical, dermoscopy)에서 2,149,706개 unlabeled skin image로 pretraining하고 28개 downstream dataset에서 평가.
- Imbalance 처리: 별도 resampling보다 대규모 self-supervised pretraining과 label-efficient 학습으로 희소 label 문제를 완화하며, TBP screening에서는 216 malignant vs 197,716 benign의 극심한 imbalance를 다룸.
- Tabular model: 독립적인 tabular model은 없고, 일부 TBP screening 실험에서 lesion measurement/metadata를 image feature와 함께 사용.
- Image model: ViT-Large visual encoder, mask regressor, CLIP-Large teacher를 사용한 masked latent alignment와 visible latent alignment 기반 self-supervised model.
- Fusion 방식: TBP, clinical, dermoscopy, dermatopathology image modality를 공통 embedding으로 통합하고, 일부 screening 실험에서는 TBP image와 measurement metadata를 결합.
- 평가 지표: AUROC, AUPR, weighted F1, balanced accuracy, sensitivity, lesion detection count, reader study accuracy를 사용.
- 평가 결과: 여러 task에서 SOTA를 달성했고, early melanoma reader study에서 평균 임상의보다 10.2% 높은 정확도, human-AI 협업에서 11% 향상, TBP screening sensitivity 0.893과 불필요한 dermoscopy 약 60.8% 감소를 보고.

## 2. SkinEHDLF a hybrid deep learning approach for accurate skin cancer classification in complex systems

- 파일: `s41598-025-98205-7.pdf`
- 출처/인용: Scientific Reports, OpenAlex 인용 수 24.

- 목표와 기여: ConvNeXt, EfficientNetV2, Swin Transformer를 결합한 SkinEHDLF로 피부암 binary 및 multi-class classification 정확도를 높이는 hybrid deep learning framework 제안.
- Dataset 정보: ISIC 2024의 3D total-body photography 기반 401,059개 skin lesion image를 사용하고 melanoma, benign lesion, noncancerous anomaly 및 BCC/SCC 등 multi-class task를 구성.
- Imbalance 처리: underrepresented class에 더 큰 weight를 주는 class-weighted loss와 rotation, crop, flip, color jitter, Gaussian noise 등 augmentation 사용.
- Tabular model: 별도 tabular model은 없고 age, gender, ethnicity 같은 dataset 특성은 설명되지만 모델 입력으로 명확히 쓰이지 않음.
- Image model: ConvNeXt가 local feature, EfficientNetV2가 scalable feature, Swin Transformer가 long-range context를 추출.
- Fusion 방식: adaptive attention-based feature fusion으로 세 backbone의 feature를 동적으로 가중 결합.
- 평가 지표: accuracy, precision, recall, F1-score, AUROC, AUC, sensitivity, specificity, MCC, FPR/FNR을 사용.
- 평가 결과: binary classification에서 AUROC 99.8%, accuracy 98.76%, multi-class classification에서 accuracy 98.6%, precision 97.9%, recall 97.3%, AUROC 99.7%를 보고.

## 3. DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses

- 파일: `s41597-024-04104-3.pdf`
- 출처/인용: Scientific Data, OpenAlex 인용 수 17.

- 목표와 기여: 기존 공개 dermoscopy dataset의 subclass 부족 문제를 보완하기 위해 40개 subclass를 갖는 대규모 dermatoscopic skin lesion dataset을 공개.
- Dataset 정보: Türkiye 3개 기관에서 2008-2021년 수집한 1,627명 환자의 12,345개 high-resolution dermatoscopic image, 5 superclass, 15 main class, 40 subclass로 구성.
- Imbalance 처리: patient-level 80/20 train/test split과 class balancing 고려를 명시하고, baseline 학습에는 augmentation을 적용.
- Tabular model: 별도 tabular model은 없고 file name, lesion class, taxonomic label을 담은 CSV metadata를 제공.
- Image model: baseline으로 ImageNet pretrained ResNet50, Xception, InceptionResNetV2를 fine-tuning.
- Fusion 방식: 단일 dermoscopic image classification dataset이라 image fusion은 없음.
- 평가 지표: weighted accuracy와 dermatologist consensus/biopsy validation을 중심으로 기술 검증.
- 평가 결과: baseline weighted accuracy는 ResNet50 0.50, Xception 0.59, InceptionResNetV2 0.58로 낮지만, 40 subclass 세분화 데이터셋 자체가 핵심 기여.

## 4. MILK10k: A Hierarchical Multimodal Imaging-Learning Toolkit for Diagnosing Pigmented and Nonpigmented Skin Cancer and its Simulators

- 파일: `1-s2.0-S0022202X25022705-main.pdf`
- 출처/인용: Journal of Investigative Dermatology, OpenAlex 인용 수 4.

- 목표와 기여: pigmented/nonpigmented skin cancer와 simulators를 함께 다루는 hierarchical multimodal image dataset과 benchmark tool을 제공.
- Dataset 정보: 5개 센터에서 수집한 5,240개 병변의 clinical close-up + dermoscopy image pair 총 10,480장, 48개 ISIC-DX diagnosis, 95.7% histopathology ground truth, age/sex/skin tone/site metadata 포함.
- Imbalance 처리: 5-fold stratified split, CrossEntropyLoss에 inverse class-frequency weighting과 label smoothing 0.01을 적용.
- Tabular model: 독립적인 tabular model은 없고 metadata는 제공되지만 baseline은 image pair 중심.
- Image model: ResNet50 backbone의 Siamese neural network로 close-up과 dermoscopy image의 penultimate feature를 추출.
- Fusion 방식: 두 image branch의 penultimate layer를 simple concatenation한 뒤 512-dimensional projection과 fully connected classifier로 분류.
- 평가 지표: recall, specificity, top-1/top-3 accuracy, hierarchical diagnosis distance metric을 사용.
- 평가 결과: 11 generic class에서 average recall 0.426, specificity 0.960, 48-class ISIC-DX에서 top-1 accuracy 53.6%, top-3 accuracy 67.7%를 보고.

## 5. Skin region images extracted from 3D total body photographs for lesion detection

- 파일: `s41597-025-05483-x.pdf`
- 출처/인용: Scientific Data, OpenAlex 인용 수 2.

- 목표와 기여: 병변 중심 crop이 아닌 주변 피부 context까지 포함한 3D total body photography 기반 lesion detection/localisation dataset을 공개.
- Dataset 정보: Barcelona와 Brisbane의 100명 참여자에서 추출한 16,954개 skin-region tile, 각 tile은 약 7 x 9 cm 피부 영역이며 suspicious lesion bounding box, anatomical location, age group, sun damage score metadata 포함.
- Imbalance 처리: 모델 학습 논문이 아니라 imbalance 알고리즘은 없고, usage note에서 benign lesion predominance와 annotation inconsistency 가능성을 주의점으로 언급.
- Tabular model: 별도 tabular model은 없지만 metadata CSV가 image별 train/test split, anatomical location, demographics, sun damage score 등을 제공.
- Image model: 새 classification model은 없고 dataset descriptor이며, detection/localisation benchmark를 위한 annotation을 제공.
- Fusion 방식: 모델 fusion은 없고 PNG image tile과 YOLO/COCO annotation, metadata를 함께 제공하는 dataset-level 결합.
- 평가 지표: manual dermatologist review와 annotation quality control 중심이며 classifier 성능 지표는 없음.
- 평가 결과: 모든 annotated tile을 dermatologist가 검토 및 수정하여 lesion detection benchmark용 공개 데이터셋으로 검증.

## 6. Automatic melanoma detection using an optimized five-stream convolutional neural network

- 파일: `s41598-025-05675-w.pdf`
- 출처/인용: Scientific Reports, OpenAlex 인용 수 1.

- 목표와 기여: hair/noise/artifact와 class imbalance 문제를 전처리하고, multi-stream CNN으로 melanoma detection 정확도를 높이는 방법 제안.
- Dataset 정보: HAM10000, ISIC 2024, ISIC 2017, ISIC 2016 dermoscopy image dataset을 사용.
- Imbalance 처리: GAN 기반 이미지 생성으로 class balance를 맞추고 hair removal, CNN denoising, enhancement, contrast adjustment를 적용.
- Tabular model: 별도 tabular model은 없음.
- Image model: ULBP-CVA, multi-block ULBP-NP, Gabor/gradient feature map을 입력으로 쓰는 optimized multi-stream CNN(OFSCNN).
- Fusion 방식: lesion color, edge, texture, local-spatial frequency, multi-oriented gradient feature stream을 CNN 안에서 결합.
- 평가 지표: accuracy, precision, recall, F1-score, 5-fold cross-validation을 사용.
- 평가 결과: HAM10000 99.8%, ISIC 2024 99.9%, ISIC 2017 99.62%, ISIC 2016 99.6% detection accuracy를 보고하며 SOTA 대비 recall 11%, precision 10%, F1 11.4% 향상을 주장.

## 7. Improving dataset transparency in dermatologic Artificial Intelligence using a dataset nutrition label

- 파일: `s41746-025-02125-9.pdf`
- 출처/인용: npj Digital Medicine, OpenAlex 인용 수 1.

- 목표와 기여: dermatology AI dataset의 bias, limitation, risk를 사용자가 빠르게 판단하도록 Dataset Nutrition Label(DNL)을 적용하는 투명성 프레임워크 제안.
- Dataset 정보: SLICE-3D/ISIC 2024 3D total body photography crop dataset을 case study로 사용하고, dataset descriptor, Kaggle metadata, curator correspondence, SME review를 활용.
- Imbalance 처리: 직접 보정하지 않고 benign/malignant class imbalance, rare diagnosis exclusion, missing skin tone documentation, low resolution, hidden proxy risk를 DNL 위험 요소로 표시.
- Tabular model: 해당 없음; 모델 개발 논문이 아니라 dataset documentation/comment 논문.
- Image model: 해당 없음; skin lesion classifier를 학습하지 않음.
- Fusion 방식: 해당 없음; DNL은 metadata, limitations, risks를 구조화해 시각적으로 요약하는 문서화 도구.
- 평가 지표: green/yellow/gray risk category와 dermatology/AI subject matter expert review로 label 정확성을 검토.
- 평가 결과: SLICE-3D의 적합한 use case와 부적합한 use case를 분리하고, darker skin tone deployment나 rare subtype diagnosis 같은 위험 적용을 경고.

## 8. New Release of User-Captured Images from the Oregon Health & Science University Melanoma MoleMapper Project

- 파일: `s41597-025-05552-1.pdf`
- 출처/인용: Scientific Data, OpenAlex 인용 수 0.

- 목표와 기여: 임상 환경이 아닌 일반 사용자의 smartphone pigmented skin lesion 사진을 공개해 consumer-captured skin image 연구 공백을 채움.
- Dataset 정보: 4,158명 참여자 metadata, 27,499개 cropped mole image, 7,305개 nearby skin patch, 1,000개 contextual zone image를 Synapse `syn51520810`로 제공.
- Imbalance 처리: unlabelled dataset이라 class imbalance 보정은 없고, self-supervised pretraining이나 image quality analysis 용도를 제안.
- Tabular model: 별도 tabular model은 없지만 onboarding age, sex at birth, melanoma history 등 participant metadata를 제공.
- Image model: 해당 없음; 모델 성능 논문이 아닌 dataset release.
- Fusion 방식: mole crop, surrounding skin patch, zone image, basic metadata를 함께 제공하는 dataset-level 구성.
- 평가 지표: HIPAA/PHI filtering, curation, data access process를 중심으로 검증하며 classification metric은 없음.
- 평가 결과: 가장 큰 공개 consumer-collected smartphone pigmented lesion dataset으로, 비임상 촬영 환경의 품질 및 편향 분석과 self-supervised pretraining에 활용 가능.

## 9. A longitudinal dataset of tile and corresponding dermoscopic images with metadata for identifying skin cancers

- 파일: `s41597-025-05880-2.pdf`
- 출처/인용: Scientific Data, OpenAlex 인용 수 0.

- 목표와 기여: 단일 병변 이미지가 아니라 같은 사람의 여러 병변, 시간 변화, dermoscopy, metadata를 함께 제공해 clinician-like context를 반영한 ML 연구 기반 마련.
- Dataset 정보: 일반 population 196명과 high-risk melanoma cohort 284명, 총 480명에서 250,162개 3D-TBP tile lesion image와 9,389개 corresponding dermoscopic image를 제공하고, 340명은 2-7회 longitudinal image 포함.
- Imbalance 처리: 데이터셋 기술 논문이라 model-level imbalance 처리는 없고, cohort 구성과 lesion/participant metadata로 subgroup 분석 가능성을 제공.
- Tabular model: 별도 tabular model은 없지만 demographics, skin cancer history, sun exposure/protection, naevus count, anatomical location 등 풍부한 tabular metadata를 제공.
- Image model: 새 classifier는 없고, VECTRA WB360의 CNN lesion detection은 tile 추출 과정에 사용됨.
- Fusion 방식: tile image, corresponding dermoscopic image, longitudinal time point, participant-level metadata를 연결하는 dataset-level multimodal fusion.
- 평가 지표: technical validation과 extraction/metadata consistency 중심이며 신규 진단 모델 성능은 보고하지 않음.
- 평가 결과: 기존 isolated dermoscopy dataset의 한계를 보완해, 병변 간 비교와 시간 변화까지 포함하는 skin cancer identification dataset을 제공.

## 10. Segmentation and Classification of Skin Cancer Diseases Based on Deep Learning: Challenges and Future Directions

- 파일: `Segmentation_and_Classification_of_Skin_Cancer_Diseases_Based_on_Deep_Learning_Challenges_and_Future_Directions.pdf`
- 출처/인용: IEEE Access, OpenAlex 인용 수 9.

- 목표와 기여: skin cancer segmentation/classification에서 CNN, U-Net, transformer, preprocessing, deployment challenge를 종합 정리하고 향후 연구 방향을 제시.
- Dataset 정보: PH2, ISIC 2016-2024, HAM10000, Derm7pt, Fitzpatrick17k 등 공개 skin disease dataset을 비교 및 검토.
- Imbalance 처리: 자체 실험은 없고, class imbalance가 ISIC 계열 dataset 성능에 영향을 줄 수 있음을 지적하며 augmentation, preprocessing, DLINEX/cross-entropy 등 문헌상 대응을 소개.
- Tabular model: 해당 없음; tabular clinical model은 주요 대상이 아님.
- Image model: CNN, U-Net, transfer learning, attention, transformer, hybrid DL segmentation/classification 모델을 review.
- Fusion 방식: review 차원에서 pretrained CNN fusion, hybrid CNN-transformer, multimodal/fusion 접근을 논의.
- 평가 지표: accuracy, Dice, IoU, sensitivity, specificity 등 문헌별 지표를 비교하지만 자체 benchmark는 없음.
- 평가 결과: diverse dataset, explainability, transferability, rare disease coverage, robust evaluation, real-world deployment가 핵심 gap이라고 결론.

## 11. AI dermatology: Reviewing the frontiers of skin cancer detection technologies

- 파일: `1-s2.0-S2950261625000196-main.pdf`
- 출처/인용: Intelligent Oncology, OpenAlex 인용 수 7.

- 목표와 기여: skin cancer detection의 AI 기술을 segmentation, classification, reinforcement learning, transformer, multimodal integration 관점에서 종합 리뷰.
- Dataset 정보: 자체 dataset은 없고 ISIC 계열 등 공개 skin cancer image dataset과 FDA 승인/임상 적용 사례를 문헌 기반으로 다룸.
- Imbalance 처리: 자체 처리 없음; data diversity, augmentation, model bias, clinical integration 한계를 review의 주요 과제로 제시.
- Tabular model: 해당 없음; tabular model은 독립 주제가 아님.
- Image model: CNN, segmentation network, transformer, XAI model, reinforcement learning 기반 기술을 폭넓게 정리.
- Fusion 방식: transformer와 multimodal technology가 diagnostic process를 정교화한다고 논의하지만 자체 fusion architecture는 없음.
- 평가 지표: 문헌별 accuracy, precision, recall, F-score 등 성능 지표를 인용.
- 평가 결과: AI가 조기 피부암 진단의 정확도와 효율을 높였지만, 데이터 다양성, 해석가능성, 임상 통합이 남은 핵심 장애물이라고 정리.

## 12. Artificial Intelligence in the Non-Invasive Detection of Melanoma

- 파일: `life-14-01602-v2.pdf`
- 출처/인용: Life, OpenAlex 인용 수 7.

- 목표와 기여: melanoma의 비침습 진단에서 AI가 clinical image, dermoscopy, RCM/OCT 등 imaging modality별로 어떻게 활용되는지 정리.
- Dataset 정보: 자체 dataset은 없고 clinical image, dermoscopic image, Fitzpatrick 17k, SkinCAP, SLICE-3D 등 다양한 공개 및 문헌 dataset을 소개.
- Imbalance 처리: 직접 처리 없음; skin type distribution, malignant/pre-malignant 비율, dataset 다양성 부족을 문헌상 한계로 언급.
- Tabular model: 해당 없음; tabular metadata가 언급되지만 별도 모델을 제안하지 않음.
- Image model: CNN, transformer, AutoML, smartphone app, dermoscopy/RCM/OCT 기반 AI 알고리즘을 review.
- Fusion 방식: 일부 문헌에서 clinical/macroscopic/dermoscopic image와 metadata 조합을 다루지만 자체 fusion model은 없음.
- 평가 지표: accuracy, sensitivity, specificity, PPV, NPV, AUC 등 문헌별 진단 지표를 표로 비교.
- 평가 결과: AI는 melanoma triage와 biopsy 필요성 판단을 보조할 가능성이 크지만, dataset 편향과 임상 검증 부족 때문에 실제 배포에는 주의가 필요하다고 정리.

## 13. FoMoSkinNet: A Dual-Stream Deep Learning Model for Early Detection and Classification of Non-Dermoscopic Skin Lesions via Focal Modulation and Local Feature Integration

- 파일: `FoMoSkinNet_A_Dual-Stream_Deep_Learning_Model_for_Early_Detection_and_Classification_of_Non-Dermoscopic_Skin_Lesions_via_Focal_Modulation_and_Local_Feature_Integration.pdf`
- 출처/인용: IEEE Access, OpenAlex 인용 수 0.

- 목표와 기여: low-quality/non-dermoscopic skin lesion image에서 local texture와 global context를 동시에 학습하는 FoMoSkinNet을 제안.
- Dataset 정보: ISIC 2024 SLICE-3D subset에서 benign 15,000장과 malignant 15,000장으로 균형화한 30,000개 image를 사용하고 80% train, 20% test, train의 10% validation split을 적용.
- Imbalance 처리: benign/malignant 15,000:15,000 balanced dataset을 구성해 binary class imbalance를 직접 완화.
- Tabular model: 별도 tabular model은 없음.
- Image model: global feature는 FocalNet, local feature는 grouped convolution, channel shuffle, multi-scale dilated depthwise convolution을 포함한 LFNet으로 추출.
- Fusion 방식: FocalNet branch와 LFNet branch의 complementary feature를 hybrid feature fusion으로 결합.
- 평가 지표: accuracy, F1-score, specificity, precision, sensitivity/recall, ROC, AUC, confusion matrix를 사용.
- 평가 결과: ISIC 2024에서 accuracy 98.85%, precision 99.46%, recall/sensitivity 98.23%, F1-score 98.84%, specificity 99.47%를 보고.

## 14. AcuSim: A Synthetic Dataset for Cervicocranial Acupuncture Points Localisation

- 파일: `s41597-025-04934-9.pdf`
- 출처/인용: Scientific Data, OpenAlex 인용 수 6.

- 목표와 기여: cervicocranial acupuncture point localization을 위한 synthetic RGB-D dataset과 자동 rendering/labeling pipeline을 제안.
- Dataset 정보: 504 synthetic anatomical model, 63,936 RGB-D image, 174 volumetric acupoint, 11,126,952 annotation으로 구성되며 피부암/피부과 진단 데이터셋은 아님.
- Imbalance 처리: class imbalance 처리보다는 synthetic model 다양화와 domain randomization에 초점.
- Tabular model: 별도 tabular model은 없음.
- Image model: VGG19 convolution layer를 feature extractor로 쓰고 coordinate regression과 acupoint name classification을 수행하는 multitask CNN.
- Fusion 방식: RGB-D image, depth, 2D bounding box, occlusion filtering, coordinate metadata를 함께 사용하는 localization pipeline.
- 평가 지표: classification accuracy, 5 mm 이내 localization 비율, coordinate RMSE/cross-entropy loss를 사용.
- 평가 결과: CNN validation에서 accuracy 99.73%, expert annotation 대비 5 mm 이내 예측 92.86%를 보고하지만 피부암 AI 관련도는 낮음.

## 15. Medical Video Generation for Disease Progression Simulation

- 파일: `2411.11943v1.pdf`
- 출처/인용: arXiv, OpenAlex 인용 수 2.

- 목표와 기여: longitudinal medical image 부족 문제를 보완하기 위해 disease progression을 prompt-controlled video로 생성하는 MVG framework를 제안.
- Dataset 정보: CheXpert Plus 223,462장, MIMIC-CXR 227,835장, Diabetic Retinopathy Detection 35,126장, ISIC 2024 401,059장, ISIC 2018 10,015장을 사용하며 피부 이미지는 검증 도메인 중 하나.
- Imbalance 처리: class imbalance보다 longitudinal progression data 부족 문제를 다루며, imbalance 보정 기법은 핵심 기여가 아님.
- Tabular model: 별도 tabular model은 없고 clinical report text를 GPT-4로 요약/recaption해 prompt로 사용.
- Image model: Stable Diffusion 기반 Progressive Image Editing(PIE), multi-round diffusion, SEINE 기반 video transition generation을 사용.
- Fusion 방식: clinical text prompt, region guide mask, initial medical image, diffusion-generated intermediate states를 결합.
- 평가 지표: CLIP-I, classification confidence score, clinician preference study, longitudinal image MAE를 사용.
- 평가 결과: skin image task에서 PIE는 confidence 0.453과 CLIP-I 0.958로 Extrapolation(0.226/0.951)과 SVD(0.201/0.886)를 상회하고, 30명 의사 user study에서 clinical plausibility를 추가 검증.

## 누락/검증 체크

- PDF 포함 여부: 15개 PDF 모두 포함.
- 개요 항목 여부: 각 논문마다 `목표와 기여`, `Dataset 정보`, `Imbalance 처리`, `Tabular model`, `Image model`, `Fusion 방식`, `평가 지표`, `평가 결과` 8개 항목 포함.
- DOI 보정: IEEE DOI의 공백 깨짐은 `10.1109/ACCESS.2026.3674219`, `10.1109/ACCESS.2025.3569170`으로 보정.
- PanDerm 보정: 제공 PDF는 arXiv v1이지만 같은 연구의 Nature Medicine published version DOI와 OpenAlex 인용 수를 별도 표기.
- 모델 논문이 아닌 dataset/review/comment 논문은 해당 없는 항목을 명시하고 이유를 한 줄로 남김.
