# Dual-Stage Segmentation and Classification Framework

- 인용 정보: Manzoor et al., 2025, `Digital Health`, https://doi.org/10.1177/20552076251351858
- 출판 유형: peer-reviewed(동료심사 논문)
- Seed 논문 인용 관계: SLICE-3D 사용; 정확한 seed citation 확인 필요
- 데이터셋: HAM10000, ISIC 2018, ISIC 2024 SLICE-3D
- 과제: skin lesion segmentation and classification(피부 병변 분할 및 분류)
- 모달리티: SLICE-3D branch(분기)에서는 image와 metadata(메타데이터)
- 추론 입력: fusion model(융합 모델)에서 image + metadata
- Strict-contract 호환성: 호환 가능 / 확인 필요

## 목표와 기여
- lesion segmentation(병변 분할) 이후 classification(분류)을 수행하는 two-phase framework(2단계 프레임워크)를 제안한다.
- HAM10000, ISIC 2018, SLICE-3D에서 segmentation/classification을 평가한다.
- ISIC 2024 SLICE-3D에서 tabular-only(표형 단독)와 image+metadata fusion(융합) 결과를 모두 보고한다.

## 불균형 처리
- balanced dataset(균형 데이터셋)과 imbalanced dataset(불균형 데이터셋)을 모두 고려하는 방법으로 명시적으로 제시한다.
- SLICE-3D를 매우 불균형하고 임상적으로 현실적인 benchmark(기준 평가 데이터셋)로 사용한다.
- 정확한 sampling(표본 추출), fold design(폴드 설계), class-weight strategy(클래스 가중치 전략)는 PDF 수준 확인이 필요하다.

## 모델
- Tabular: XGBoost classifier(분류기).
- Image: segmentation에는 VGG16 encoder(인코더)를 포함한 U-Net, SLICE fusion에는 ResNet 기반 classifier.
- Fusion: ResNet 기반 classifier를 사용한 image + metadata fusion(융합).

## 지표와 결과
- 지표: SLICE-3D에는 pAUC; 다른 데이터셋에는 accuracy(정확도), F1, sensitivity(민감도), specificity(특이도), Jaccard, Dice.
- 최고 보고 SLICE 결과: tabular-only XGBoost pAUC 0.16752.
- Image+tabular 결과: ResNet fusion(융합) 사용 시 pAUC 0.15792.
- 검증/테스트 protocol(절차): 확인 필요.
- Threshold(임계값) 선택: 확인 필요.

## 한계
- 보고된 SLICE 결과에서는 fusion(융합)이 tabular-only보다 낮아 architecture(구조) 또는 protocol(절차)이 약할 수 있다.
- 데이터셋마다 다른 metric(지표)을 사용하므로 비교를 과장하기 쉽다.
- 논문 인용 전에 patient-level split(환자 단위 분할)과 train-only preprocessing(학습 데이터만으로 전처리) 검증이 필요하다.

## 우리 연구와의 관련성
- image+tabular fusion(융합)이 자동으로 tabular-only보다 낫지는 않다는 중요한 contrast case(대조 사례)이다.
- 같은 fold(폴드) 아래에서 image-only(영상 단독), tabular-only(표형 단독), multimodal(멀티모달)을 비교해야 한다는 우리 계획을 뒷받침한다.
- fold-wise ablation(폴드별 구성요소 제거 비교) 없이 multimodal 이득을 주장하지 말아야 한다는 경고로 유용하다.

## 검증 메모
- 출처: SAGE/Digital Health abstract와 result snippet.
