# ISIC 2024 기반 멀티모달 피부암 탐지 논문 조사

논문별 분석: 목표, 불균형 처리, 모델 구성, 결합 방식, 지표와 성능

| 항목 | 내용 |
| --- | --- |
| 조사 범위 | 2023-04-29 이후 발표된 ISIC 2024/SLICE-3D 기반 이미지+tabular(metadata) 멀티모달 연구 |
| 기준 논문 | SLICE-3D dataset 논문(Scientific Data, 2024)|

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
