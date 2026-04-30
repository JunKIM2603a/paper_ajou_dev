# ISIC 2024 기반 멀티모달 피부암 탐지 논문 조사

논문별 분석: 목표, 불균형 처리, 모델 구성, 결합 방식, 지표와 성능

| 항목 | 내용 |
| --- | --- |
| 조사 범위 | 2023-04-29 이후 발표된 ISIC 2024/SLICE-3D 기반 이미지+tabular(metadata) 멀티모달 연구 |
| 기준 논문 | SLICE-3D dataset 논문(Scientific Data, 2024)|
| 구성 방식 | 비교표 대신 논문별 섹션으로 정리. 각 논문마다 요청된 7개 분석 항목을 같은 순서로 배치 |
| 각주 방식 | 논문별로 생소한 용어를 본문 직후 ‘용어 각주’에 일반적인 말로 풀이 |

> SLICE-3D 논문은 3D 전신 촬영에서 추출한 40만 개 이상의 피부 병변 crop과 metadata를 공개한 데이터셋 논문으로, ISIC 2024의 기반이 되었고, dermoscopy 중심 데이터셋의 selection bias를 줄여 실제 triage 환경에 가까운 피부암 AI 연구를 가능하게 한 자료다. 

## 핵심 결론
1. 극심한 Class Imbalance 문제
> ISIC 2024/SLICE-3D 연구의 핵심 난점은 malignant 비율이 약 0.1% 수준으로 낮은 극심한 class imbalance이다.
  
2. Image + Metadata 기반 Late Fusion 전략
> 고성능 접근은 image model 단독보다 image prediction/embedding을 metadata와 결합하는 late fusion 또는 GBDT ensemble을 선호한다.
  
3. Patient-level Context 기반 병변 해석
> patient-level context, lesion outlier feature, WB360 appearance metadata처럼 ‘한 병변을 환자 전체 병변 속에서 보는 정보’가 중요한 성능 요인이다.

4. High Sensitivity 중심 평가 지표
> 평가는 high sensitivity 영역을 강조하는 pAUC >80% TPR이 중심이며, AUC/F1/recall/top-15 retrieval sensitivity가 보조 지표로 사용된다.

<br>

## 0. The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection

Scientific Data, 2024 | ISIC 2024 challenge 기반 데이터셋 논문 | [원문 링크](https://www.nature.com/articles/s41597-024-03743-w)

요약: SLICE-3D는 3D total-body photography(3D TBP)에서 개별 피부 병변을 crop한 400,000개 이상의 non-dermoscopic tile image와 metadata를 공개한 데이터셋 논문이다. 기존 공개 피부암 데이터셋이 dermoscopy 중심이고 수상한 병변 위주로 모이는 selection bias를 갖는다는 한계를 줄이기 위해, 한 환자에게 존재하는 많은 병변을 체계적으로 추출해 실제 triage 환경에 가까운 연구 기반을 제공한다.

### 1. 데이터 구성과 기술 검증 방식

데이터셋은 개별 병변 tile image와 metadata record로 구성된다. 각 tile은 15 mm x 15 mm field-of-view로 추출되며, 평균 이미지 크기는 약 133 x 133 px이다. metadata에는 patient sex/age, anatomical location, lesion size, 3D coordinate, lighting/tile type, Lesion Visualizer 기반 색·모양·크기 feature, patient_id 등이 포함된다.

### 2. 불균형과 라벨 특성

SLICE-3D는 실제 triage 환경을 반영하기 때문에 malignant 병변이 매우 적고 benign 병변이 압도적으로 많다. 

또한 malignant label은 병리 근거가 강하지만, benign label은 weak-label이 많다. 따라서 SLICE-3D 기반 모델은 malignant detection 관점에서 유용하지만, benign 전체를 완전한 병리 음성으로 해석하면 안 된다.

- strong-label: 조직병리학으로 확인된 Malignant class
- weak-label: 임상적으로 benign으로 간주된 병변

### 7. 핵심 기여와 결론

SLICE-3D는 dermoscopy 중심 연구를 넘어 **3D TBP 기반 non-dermoscopic triage 환경**을 제공하고, **patient_id와 WB360 metadata를 통해 patient-level context 기반 분석을 가능하게 한 데이터셋**이다.

특히 논문은 benign moles가 한 환자 안에서 서로 닮는 경향이 있고, 그중 튀는 병변이 melanoma일 가능성이 높다는 ugly duckling sign을 언급한다. SLICE-3D는 한 환자의 병변 phenotype을 더 완전하게 표현하므로, 이런 patient-level outlier feature를 만드는 후속 연구의 기반이 된다.

<br>

## 1. Automated triage of cancer-suspicious skin lesions with 3D total-body photography

npj Digital Medicine, 2025 | ISIC 2024 challenge 결과·benchmark 논문원문 링크

요약: ISIC 2024 1등 solution을 분석한 benchmark 연구로, **RandomOverSampler·RandomUnderSampler·scale_pos_weight**로 class imbalance를 다루고, **image prediction + metadata + patient-context feature**를 결합한 **late fusion GBDT ensemble**이 효과적인 피부암 triage 성능을 보였음을 제시했다.

### 0. 파이프라인

아래 그림은 1등 solution을 최종 GBDT 관점에서 정리한 흐름이다. 이미지 모델은 별도로 학습되지만, 최종 GBDT 입장에서는 이미지를 확률 컬럼으로 바꾸는 feature generator 역할을 한다.

```text
┌────────────────────────────────────────────────────────┐
│                       [RAW DATA]                       │
│ - lesion crop image                                    │
│ - metadata / WB360 appearance features                 │
│ - patient_id                                           │
│ - target                                               │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ patient_id-based
                            │ StratifiedGroupKFold
                            ▼
┌────────────────────────────────────────────────────────┐
│            [IMAGE MODEL TRAINING PER FOLD]             │
│                                                        │
│  [ train patients: image + target ]                    │
│                   │                                    │
│                   ▼                                    │
│  [ EVA02 / EdgeNeXt image classifiers ]                │
│                   │                                    │
│                   ▼                                    │
│  [ validation patients: OOF malignant probability ]    │
└───────────────────────────┬────────────────────────────┘
                            │ -> predictions_eva
                            │ -> predictions_edg
                            ▼
┌────────────────────────────────────────────────────────┐
│             [MULTIMODAL TABULAR DATASET]               │
│ - metadata columns                                     │
│ - engineered feature columns                           │
│ - patient-context feature columns                      │
│ - predictions_eva                                      │
│ - predictions_edg                                      │
│ - target                                               │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ over/under-sampling applied here
                            │ (row-level sampling, 
                            │  not image-file sampling)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 [FINAL GBDT TRAINING]                  │
│             LightGBM / XGBoost / CatBoost              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             ensemble malignant risk score              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              evaluation: pAUC, AUC, NNT                │
└────────────────────────────────────────────────────────┘
```

### 1. 논문의 목표와 핵심 기여

ISIC 2024 challenge 결과와 1등 solution을 분석한 benchmark 연구로, 3D total-body photography에서 자동 추출된 많은 피부 병변 중 암이 의심되는 병변을 AI가 먼저 선별하는 triage system을 검증한 연구이다.

### 2. 핵심 문제

malignant lesion이 약 393개, benign lesion이 약 400,666개로 negative:positive 비율이 약 1019:1인 극심한 class imbalance가 존재한다. 따라서 단순 accuracy보다 암을 충분히 잡는 high-sensitivity 조건의 성능이 중요하다.

### 3. 평가 방식

주요 평가지표는 pAUC >80% TPR이며, NNT 80% SE도 함께 해석한다. 이는 암 병변을 최소 80% 이상 잡는 조건에서 모델이 얼마나 효율적으로 의심 병변을 줄여주는지 보기 위한 지표다.
winning model pAUC >80% TPR 0.1726, AUC 0.9668, NNT 80% SE 51.57, NNT 90% SE 98.20, top-15 patient-level sensitivity 0.7903 (patient-context 제거 시 AUC 0.967→0.956 및 NNT 악화)


### 4. 모델 구조

전체 구조는 multimodal late fusion pipeline이다. 먼저 EVA02-small, EdgeNeXt 같은 image model이 병변 crop image에서 malignant probability를 만들고, 이 값을 metadata와 함께 tabular feature로 사용한다.

### 5. 최종 예측 모델

최종 decision maker는 LightGBM, XGBoost, CatBoost로 구성된 GBDT ensemble이다. 이 모델은 image prediction, demographic metadata, WB360 appearance metadata, engineered feature, patient-context feature를 함께 사용해 최종 risk score를 산출한다.

### 6. 불균형 처리

RandomOverSampler, RandomUnderSampler, scale_pos_weight를 사용했다. 단, sampling은 원본 이미지 파일이 아니라 image prediction이 포함된 lesion-level tabular row에 적용된다.

### 7. 핵심 기여와 결론

feature ablation study를 통해 image, metadata, patient-context feature의 기여도를 분석했고, image-only 방식보다 image prediction을 metadata 및 환자 내 context와 결합한 late fusion 방식이 triage 성능을 더 높인다는 점을 보였다.

<br>

## 2. Hybrid Ensemble of Segmentation-Assisted Classification and GBDT

arXiv, 2025 | ISIC 2024/SLICE-3D 기반 hybrid late-fusion 접근 | [원문 링크](https://arxiv.org/abs/2506.03420)

**image model이 만든 malignant probability를 metadata, engineered feature, patient-specific relational feature와 결합한 뒤 GBDT ensemble로 최종 판단**하고, 여기에 segmentation-assisted image classifier, synthetic malignant lesion, external dataset relabeling을 추가해 pAUC >80% TPR을 끌어올렸다.

### 0. 파이프라인

```text
┌───────────────────────────────────────────────────────────────────────────┐
│             [ SLICE-3D lesion crop + metadata + patient_id ]              │
└─────────┬───────────────────────────┬───────────────────────────┬─────────┘
          │                           │                           │
┌─────────▼─────────┐       ┌─────────▼─────────┐       ┌─────────▼─────────┐
│   Image Branch    │       │  Tabular Branch   │       │  Patient Branch   │
│ EVA02/EdgeNeXtSAC │       │    metadata +     │       │  patient-context  │
│                   │       │engineered feature │       │      branch       │
└─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
          │                           │                           │
┌─────────▼─────────┐       ┌─────────▼─────────┐       ┌─────────▼─────────┐
│ malignant prob /  │       │   size, color,    │       │ pt-norm. ratio,   │
│  softmax feature  │       │ geometry, spatial │       │  outlier score,   │
│                   │       │      feature      │       │ relational feat.  │
└─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                [ LightGBM + XGBoost + CatBoost ensemble ]                 │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      [ final malignant risk score ]                       │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1. 논문의 목표와 핵심 기여

segmentation-assisted image model과 metadata·patient-context를 결합한 hybrid late fusion GBDT로 image-only 한계를 극복.

### 2. 핵심 문제

극심한 class imbalance와 non-dermoscopic 이미지의 낮은 경계·texture 정보를 segmentation-assisted classification을 통해 모델이 배경보다 병변 영역에 더 집중하도록 설계

### 3. 평가 방식

pAUC >80% TPR 중심으로 구성 요소별 성능 기여를 비교 분석.

### 4. 모델 구조

* image model(EVA02, EdgeNeXtSAC) → malignant probability 생성 → tabular feature 구성 → GBDT ensemble로 risk score 산출.
(401,059개 crop, 393 real malignant + 30,228 synthetic malignant, 214개 GBDT input feature,3x5-fold setup, 45개 GBDT ensemble, EdgeNeXtSAC의 dual-head/UNet-like decoder/CBAM/loss 구성,image model별 pAUC ablation)

* EdgeNeXtSAC: 학습 중 segmentation mask와 CAM/segmentation loss를 이용해 병변 부위에 집중하도록 훈련되고, 학습 후에는 classification head의 malignant probability를 predictions_edg feature로 추출해 tabular feature와 함께 GBDT ensemble에 입력한다.
  * lesion probability map: EdgeNeXtSAC를 병변 중심으로 학습시키기 위한 보조 출력
  * predictions_edg: 학습된 EdgeNeXtSAC가 만든 malignant 확률값, 최종 GBDT에 들어가는 image-derived feature
```text
┌────────────────────────────────────────────────────────────┐
│                       [ INPUT DATA ]                       │
│                        lesion image                        │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                       [ EdgeNeXtSAC ]                      │
│                   shared EdgeNeXt encoder                  │
└───────────────┬────────────────────────────┬───────────────┘
                │                            │
┌───────────────▼──────────────┐ ┌───────────▼───────────────┐
│    [ Classification Head ]   │ │  [ Segmentation Decoder ] │
│ malignant logit / probability│ │   lesion probability map  │
└───────────────┬──────────────┘ └───────────┬───────────────┘
                │                            │
                │                            ▼
                │                ┌───────────────────────────┐
                │                │ compare with ground-truth │
                │                │     segmentation mask     │
                │                └───────────┬───────────────┘
                ▼                            ▼
┌──────────────────────────────┐ ┌───────────────────────────┐
│   classification BCE loss    │ │     segmentation loss     │
│                              │ │      + CAM Dice loss      │
└───────────────┬──────────────┘ └───────────┬───────────────┘
                │                            │
                └─────────────┬──────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                   [ TRAINING OBJECTIVE ]                   │
│                                                            │
│ total loss = classification BCE loss                       │
│            + CAM Dice loss                                 │
│            + segmentation BCE/Dice loss                    │
└────────────────────────────────────────────────────────────┘


==============================================================
                      [ AFTER TRAINING ]
==============================================================


┌────────────────────────────────────────────────────────────┐
│             [ INFERENCE / FEATURE EXTRACTION ]             │
│                        lesion image                        │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    trained EdgeNeXtSAC                     │
│                (classification head output)                │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                      predictions_edg                       │
│           = malignant probability / class score            │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│               combined with tabular features               │
│        (metadata + engineered + patient-specific)          │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                       GBDT ensemble                        │
│              (LightGBM / XGBoost / CatBoost)               │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                 final malignant risk score                 │
└────────────────────────────────────────────────────────────┘
```

### 5. 최종 예측 모델

LightGBM·XGBoost·CatBoost GBDT ensemble이 최종 decision 수행.
image model은 최종 classifier라기보다 probability feature generator.
GBDT는 metadata, engineered feature, patient-specific relational feature, image probability를 함께 보고 benign/malignant risk를 계산(late fusion 계열)

### 6. 불균형 처리

불균형 처리는 세 방향이다. 
1. Stable Diffusion으로 synthetic malignant lesion을 생성해 malignant class의 시각적 예시를 늘린다. 
2. 외부 데이터셋의 다양한 진단명을 nevus, melanoma, bkl 중심의 3-class 형태로 relabeling하여 보조 학습에 활용한다. 
3. balanced sampling을 사용해 학습 시 benign class가 압도하지 않도록 한다.

### 7. 핵심 기여와 결론

late fusion 구조와 patient-context feature 결합이 핵심이며 pAUC 0.1755로 성능 향상.

<br>

## 6. Multimodal system for skin cancer detection

System Research and Information Technologies, 2026 | ISIC 2024 main data 기반 conventional-photo + metadata multimodal system | [원문 링크](https://journal.iasa.kpi.ua/article/view/358061)

요약: 이 논문은 ISIC 2024/SLICE-3D challenge data를 main data로 사용하고, ISIC Archive와 Stable Diffusion generated data를 보조 데이터로 활용한 multimodal melanoma detection system이다. 핵심은 image model, metadata/tabular model, image prediction을 다시 tabular model에 넣는 two-stage 구조, 그리고 여러 모델 출력을 Optuna coefficient optimization으로 합치는 three-stage ensemble이다. 불균형이 극심한 ISIC 2024 조건에서 patient/body-region aggregate feature, balanced/square sampling, RandomUnderSampler, ASL/Focal loss, heavy augmentation을 조합해 성능을 높인다.

| 분석 항목 | 논문 내용 |
| --- | --- |
| 목표와 핵심 기여 | ISIC 2024/SLICE-3D를 main data로 사용해 conventional-photo-like lesion crop과 metadata를 결합하는 scalable multimodal melanoma detection system 제안. metadata가 있는 경우와 없는 경우 모두 대응하는 two-step strategy를 구성. |
| Imbalanced data 처리 | ISIC Archive 71,080장과 Stable Diffusion generated data 6,012장을 보조 데이터로 사용. Neural model은 balanced sampling, square balancing, heavy augmentation, BCE/Focal/ASL loss, TTA를 사용. Tabular model은 RandomUnderSampler와 Optuna tuning을 사용. |
| Tabular model | LightGBM/XGBoost 기반 boosting model. 기본 metadata에 spatial/color/physical engineered feature와 patient/body-region aggregate feature를 추가. Aggregated feature가 tabular 성능을 크게 높임. |
| Image model | ConvNeXt V2 Pico, EdgeNeXt Base, EfficientNetV2 B0/B2 계열 CNN encoder 사용. timm pretrained model을 시작점으로 image-only 또는 image+tabular neural model을 학습. |
| Tabular-image 결합 | Multi-modal neural network에서 image embedding과 feed-forward tabular embedding을 결합. 이후 image-only prediction을 tabular feature로 넣는 second-stage model과 최종 three-stage ensemble을 구성. |
| 평가 지표 | Partial ROC AUC, full ROC AUC, top-15 retrieval sensitivity, public/private leaderboard score. pAUC는 TPR >80% 구간 성능을 강조하는 ISIC 2024 중심 지표. |
| 최종 결과/성능 | Three-stage v2: validation partial ROC AUC 0.18068, private 0.17042, public 0.18528, top-15 sensitivity 0.78371. ConvNeXt+ASL multimodal private 0.16458. |

### 1. 논문의 목표와 핵심 기여

이 논문은 ISIC 2024/SLICE-3D challenge dataset을 main data로 사용해, conventional photo에 가까운 3D-TBP lesion crop과 metadata를 결합하는 melanoma detection system을 제안한다. Dermoscopy처럼 전문 장비에 의존하는 환경보다, telehealth나 일반 임상 환경에 가까운 낮은 품질의 close-up photo에서도 동작 가능한 system을 목표로 한다.

핵심 기여는 세 가지다. 첫째, image data와 tabular metadata를 함께 처리하는 multimodal neural network를 만든다. 둘째, metadata가 없는 외부 image data도 활용할 수 있도록 two-step training strategy를 사용한다. 셋째, tabular model, multimodal model, image-only prediction을 활용한 second-stage model을 최종 three-stage ensemble로 결합한다.

### 2. 핵심 문제

ISIC 2024 main data는 401,059개 tile 중 400,666개가 benign이고 393개만 malignant인 극단적 class imbalance를 가진다. 또한 image는 dermoscopy보다 낮은 품질의 3D-TBP crop으로, 스마트폰 close-up photo와 비슷한 특성을 갖는다. 라벨도 strong-label과 weak-label이 섞여 있어 label confidence가 균일하지 않다.

이 문제를 해결하기 위해 논문은 ISIC 2024만 단독으로 쓰지 않는다. ISIC Archive의 malignant image를 보조 데이터로 사용하고, Stable Diffusion 기반 generated image도 추가한다. 단, ISIC 2024가 main data이며, 외부/생성 데이터는 malignant 부족과 image model robustness를 보완하기 위한 보조 데이터로 사용된다.

### 3. 평가 방식

주요 평가지표는 Partial ROC AUC, ROC AUC, top-15 retrieval sensitivity다. Partial ROC AUC는 TPR >80% 구간의 ROC area를 계산하며, 0.0에서 0.2 사이 값을 갖는다. ISIC 2024처럼 malignant를 놓치지 않는 high-sensitivity triage가 중요한 문제에서 핵심 지표로 사용된다.

Top-15 retrieval sensitivity는 환자 또는 평가 단위에서 가장 의심스러운 상위 후보 안에 실제 malignant가 얼마나 들어오는지를 보는 지표다. 논문은 validation score뿐 아니라 Kaggle public/private leaderboard score도 함께 사용한다. 특히 two-stage/three-stage validation은 낙관적일 수 있다고 보고, public/private score를 별도로 제시한다.

### 4. 모델 구조

모델은 크게 three-stage system으로 이해할 수 있다.

Stage 1에서는 여러 1차 모델을 학습한다. 여기에는 tabular model, image+tabular multimodal neural model, image-only neural model이 포함된다. Image branch는 ConvNeXt V2 Pico, EdgeNeXt Base, EfficientNetV2 계열 encoder를 사용하고, tabular branch는 feed-forward neural network 또는 LightGBM/XGBoost 기반 boosting model을 사용한다.

Stage 2에서는 image-only model의 prediction을 tabular feature에 추가해 새로운 tabular model을 학습한다. 즉 image model output을 `Automated triage`나 `Hybrid Ensemble`처럼 tabular feature로 바꾸어 LightGBM/XGBoost 계열 model이 다시 판단하게 한다.

Stage 3에서는 Stage 1과 Stage 2의 여러 model output을 ensemble한다. 서로 다른 모델의 probability distribution이 다르기 때문에 rank method로 prediction을 표준화하고, Optuna coefficient optimization으로 ensemble weight를 찾는다. Overfitting을 줄이기 위해 여러 optimization run의 top result를 평균하고, 최종 weight를 수동 조정한다.

### 5. 최종 예측 모델

최종 예측은 단일 CNN이나 단일 GBDT가 아니라 three-stage ensemble이다. 1차 모델들은 image-only, image+tabular, tabular-only로 각각 다른 정보를 학습하고, 2차 모델은 image-only prediction을 tabular feature로 받아 보정한다. 마지막 3차 ensemble은 이 출력들을 rank-standardization 후 가중 결합한다.

성능은 three-stage v2가 가장 강하다. 논문은 three-stage v2가 validation partial ROC AUC 0.18068, private score 0.17042, public score 0.18528, top-15 retrieval sensitivity 0.78371을 기록했다고 보고한다. 단일 multimodal ConvNeXt 모델은 partial ROC AUC mean 0.17698, private 0.16090, public 0.17714였고, ConvNeXt+ASL loss 구성은 private 0.16458로 더 좋은 private 결과를 보였다.

### 6. 불균형 처리

불균형 처리는 세 층으로 이루어진다.

첫째, 데이터 보강이다. ISIC 2024 main data의 malignant가 393개뿐이므로, 논문은 ISIC Archive를 필터링해 71,080장 이미지를 보조 데이터로 사용한다. 이 중 malignant는 9,170장으로, ISIC 2024보다 훨씬 많은 malignant example을 제공한다. 또한 Stable Diffusion 2 기반 generated data 6,012장도 사용하며, 이 데이터는 malignant 3,012장과 benign 3,000장으로 거의 균형을 이룬다.

둘째, neural model 학습에서 balanced sampling과 square balancing을 쓴다. 1단계에서는 positive/negative가 균등하게 보이도록 balanced sampling을 사용하고, 2단계에서는 class weight를 제곱근 형태로 조정하는 square balancing을 적용한다. Heavy augmentation도 사용한다. Transpose, vertical/horizontal flip, brightness/contrast, blur/noise, distortion, CLAHE, hue/saturation/value shift, shift/scale/rotation, coarse dropout을 적용해 malignant 부족과 overfitting을 완화한다. Loss는 BCE를 기본으로 하며, Focal loss와 ASL loss도 ablation에서 비교한다.

셋째, tabular model 학습에서는 모든 tabular model 앞단에 RandomUnderSampler를 사용한다. LightGBM/XGBoost model의 under-sampling ratio, epoch 수, early stopping 여부, hyperparameter는 Optuna로 튜닝한다. Early stopping을 쓰는 경우 5개 모델 ensemble을 만들고, 각 모델은 4/5 data로 학습하고 1/5 data로 early stopping을 수행한다.

### 7. 핵심 기여와 결론

이 논문의 핵심 기여는 ISIC 2024/SLICE-3D를 main data로 사용하는 multimodal system에서 image, metadata, patient/body-region aggregate feature, external/generated data, multi-stage ensemble을 모두 결합했다는 점이다. 단순 image-only model보다 metadata 결합이 강하고, 단순 tabular model보다 image prediction과 ensemble을 함께 쓰는 multi-stage system이 더 높은 성능을 보인다.

특히 patient/body-region aggregate feature의 효과가 중요하다. Tabular ablation에서 basic LightGBM은 mean partial ROC AUC 0.1586이었고, additional spatial/color/physical feature를 넣으면 0.1604로 소폭 상승했다. 하지만 같은 환자 또는 같은 body region 안에서 lesion characteristic을 비교하는 aggregated feature를 넣으면 0.1728까지 상승했다. 이는 `Automated triage`의 patient-context feature와 같은 방향의 결과다.

또한 image+tabular multimodal model은 image-only보다 크게 개선된다. 논문은 모든 multimodal setup이 image-only보다 상당히 높은 성능을 보였고, 그 이유를 tabular feature가 image보다 noise가 적기 때문이라고 설명한다. 따라서 이 논문은 “이미지 모델을 잘 만드는 것”뿐 아니라 “metadata와 patient/body-region context를 어떻게 결합할 것인가”가 ISIC 2024 triage에서 중요하다는 점을 보여준다.

### 파이프라인으로 이해하기

```text
DATA SOURCES
  |
  |-- Main data: ISIC 2024 / SLICE-3D
  |      - 401,059 tiles
  |      - 393 malignant / 400,666 benign
  |      - image + metadata
  |
  |-- Auxiliary data: ISIC Archive
  |      - 71,080 filtered images
  |      - more malignant examples
  |
  |-- Auxiliary data: Generated Data
         - 6,012 Stable Diffusion images
         - 3,012 malignant / 3,000 benign

STAGE 1: first-level models
  |
  |-- image-only CNN
  |      ConvNeXt / EdgeNeXt / EfficientNetV2
  |
  |-- image + tabular neural model
  |      image embedding + feed-forward metadata embedding
  |
  |-- tabular model
         LightGBM / XGBoost with engineered + aggregate features

STAGE 2: prediction-as-feature
  |
  image-only prediction
        +
  tabular features
        |
        v
  second-stage LightGBM / XGBoost

STAGE 3: final ensemble
  |
  outputs from Stage 1 and Stage 2
        |
        v
  rank standardization
        |
        v
  Optuna coefficient optimization
        |
        v
  final melanoma risk score
```

### 용어 설명

- `Main Data`: 이 논문에서는 ISIC 2024 Kaggle Challenge dataset, 즉 SLICE-3D 기반 image+metadata 데이터를 뜻한다.
- `balanced sampling`: positive와 negative class가 학습 중 더 균등하게 보이도록 sample을 뽑는 방식이다.
- `square balancing`: class 빈도의 역수를 그대로 쓰지 않고 제곱근 형태로 완화해 class weight를 주는 방식이다.
- `RandomUnderSampler`: 다수 class인 benign sample을 줄여 tabular model 학습을 안정화하는 imbalanced-learn sampler다.
- `rank standardization`: 모델마다 probability 분포가 다르기 때문에, 확률값을 rank로 바꾼 뒤 ensemble하는 방식이다.
- `top-15 retrieval sensitivity`: 상위 15개 의심 후보 안에 실제 malignant가 얼마나 포함되는지 보는 triage 지표다.

### Automated triage와의 비교

`Automated triage`와 이 논문은 모두 ISIC 2024/SLICE-3D의 극단적 class imbalance와 patient-level context의 중요성을 다룬다. `Automated triage`는 WB360 metadata, image prediction, patient-context feature를 GBDT ensemble에 결합하고 그 ablation을 분석한다. 이 논문은 patient/body-region aggregate feature와 multi-stage ensemble을 통해 유사한 방향의 정보를 사용한다.

`Hybrid Ensemble`과도 유사하다. 두 논문 모두 image prediction을 tabular model에 feature로 넣는 late-fusion 구조를 사용한다. 차이는 이 논문이 ISIC Archive와 generated data를 더 적극적으로 사용하고, image+tabular neural model과 three-stage ensemble까지 포함한 더 넓은 system-level 구성을 제시한다는 점이다.

<br>

## 종합 분석

Late fusion이 우세하다. image encoder가 직접 최종 판단을 내리기보다, image prediction 또는 embedding을 tabular feature로 바꾸고 GBDT 계열 모델이 metadata와 함께 최종 risk score를 산출한다.

Tabular metadata의 영향이 크다. lesion size, color contrast, anatomical location, age/sex, 병원/site label, patient-wise normalized feature가 image-only 성능을 보완한다.

불균형 대응은 모델 성능의 핵심이다. malignant sample이 매우 적기 때문에 synthetic lesion generation, augmentation, balanced sampling, under-sampling, class-specific loss, external dataset relabeling이 반복적으로 사용된다.

임상 적용을 위해 설명가능성이 중요해지고 있다. SHAP/CAM/Grad-CAM, feature importance, nomogram, patient-level rank 같은 방식으로 모델 판단 근거를 제시하려는 흐름이 강하다.

## 한계 및 발표 시 유의점

- Automated triage 논문은 공개 시점이 늦어 직접 후속 인용 논문 수가 제한적이다. 발표에서는 ‘후속 인용 논문’보다 ‘ISIC 2024 benchmark/context 논문’으로 설명하는 것이 안전하다.

- Kaggle challenge 기반 연구는 public leaderboard 반복 제출로 인한 overfitting 가능성이 있으므로 private score와 외부 검증 성능을 구분해야 한다.

- WB360 appearance metadata는 proprietary tooling에서 산출되므로 다른 촬영 시스템이나 스마트폰 환경에 그대로 적용하기 어렵다.

- 여러 논문이 internal validation 또는 challenge score 중심이어서 실제 병원 workflow 적용 전 prospective external validation이 필요하다.

## 참고문헌 및 근거 링크

[1] Kurtansky et al. The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection. Scientific Data, 2024. [링크](https://www.nature.com/articles/s41597-024-03743-w)

[2] Kurtansky et al. Automated triage of cancer-suspicious skin lesions with 3D total-body photography. npj Digital Medicine, 2025. [링크](https://www.nature.com/articles/s41746-025-02070-7)

[3] Hasan & Rifat. Hybrid Ensemble of Segmentation-Assisted Classification and GBDT for Skin Cancer Detection with Engineered Metadata and Synthetic Lesions from ISIC 2024 Non-Dermoscopic 3D-TBP Images. arXiv:2506.03420, 2025. [링크](https://arxiv.org/abs/2506.03420)

[4] Manzoor et al. Dual-stage segmentation and classification framework for skin lesion analysis using deep neural network. Digital Health, 2025. [링크](https://journals.sagepub.com/doi/full/10.1177/20552076251351858)

[5] Wang et al. Explainable multimodal AI for skin lesion risk prediction via 3D imaging and clinical data. Scientific Reports, 2025. [링크](https://www.nature.com/articles/s41598-025-33536-z)

[6] Wu & Leng. Anti-Occlusion Diagnosis of Skin Cancer Based on Heterogeneous Data. Journal of Multimedia Information System, 2025. [링크](https://www.jmis.org/archive/view_article?pid=jmis-12-2-51)

[7] Sydorskyi et al. Multimodal system for skin cancer detection. System Research and Information Technologies, 2026. [링크](https://journal.iasa.kpi.ua/article/view/358061)
