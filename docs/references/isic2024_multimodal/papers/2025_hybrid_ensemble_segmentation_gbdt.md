# Hybrid Ensemble of Segmentation-Assisted Classification and GBDT

- 인용 정보: Hasan and Rifat, 2025, arXiv:2506.03420, https://doi.org/10.48550/arXiv.2506.03420
- 출판 유형: preprint(사전공개 논문)
- Seed 논문 인용 관계: 직접 SLICE-3D 사용; 정확한 citation relation 확인 필요
- 데이터셋: ISIC 2024 SLICE-3D와 external harmonized data(외부 조화 데이터)
- 과제: malignant vs benign skin lesion classification
- 모달리티: image, engineered metadata(설계 메타데이터), patient-specific relational metric(환자별 관계 지표)
- 추론 입력: image + engineered metadata; training/data preparation(학습/데이터 준비)에서 diagnosis-informed external relabeling(진단 정보 기반 외부 데이터 재라벨링) 사용
- Strict-contract 호환성: 부분 호환

## 목표와 기여
- ISIC 2024 non-dermoscopic(비더모스코피) 3D-TBP image를 위한 hybrid machine/deep learning system(하이브리드 머신러닝/딥러닝 시스템)을 제안한다.
- segmentation-assisted image classification(분할 보조 영상 분류)을 GBDT metadata ensemble(메타데이터 앙상블)과 결합한다.
- synthetic malignant lesion(합성 악성 병변)과 external dataset(외부 데이터셋)의 diagnosis-informed harmonization(진단 정보 기반 조화)을 추가한다.

## 불균형 처리
- Stable Diffusion으로 생성한 synthetic lesion(합성 병변)으로 malignant case(악성 사례)를 증강한다.
- diagnosis-informed relabeling(진단 정보 기반 재라벨링)을 사용해 external data를 단순화된 three-class setup(3클래스 설정)으로 mapping(매핑)한다.
- 이 relabeling은 유용할 수 있지만 strict baseline(엄격 기준 모델)이 아니라 candidate/external-data logic(후보/외부 데이터 로직)으로 다뤄야 한다.

## 모델
- Tabular: engineered feature(설계 특징)와 patient relational metric을 사용하는 GBDT ensemble.
- Image: EVA02와 EdgeNeXtSAC.
- Fusion: image prediction(영상 예측)을 GBDT ensemble 안에서 metadata feature와 결합한다.

## 지표와 결과
- 지표: pAUC above 80% TPR.
- 최고 보고 결과: pAUC 0.1755.
- 검증/테스트 protocol(절차): 논문은 configuration(설정) 중 최고라고 설명함; 정확한 split detail(분할 세부 사항) 확인 필요.
- Threshold(임계값) 선택: abstract에는 명시되지 않음; 확인 필요.

## 한계
- full peer-review signal(완전한 동료심사 신호)이 없는 preprint이다.
- Diagnosis-informed relabeling은 분리하지 않으면 strict inference/data-contract framing(엄격 추론/데이터 계약 구성)과 충돌할 수 있다.
- Synthetic data effect(합성 데이터 효과)는 paper claim(논문 주장)에 사용하기 전에 별도 ablation이 필요하다.

## 우리 연구와의 관련성
- hybrid fusion(하이브리드 융합), metadata engineering(메타데이터 설계), synthetic positive augmentation(합성 양성 증강)에 유용한 reference이다.
- strict baseline이 안정화된 뒤 별도의 imbalance/synthetic-data ablation(불균형/합성 데이터 제거 비교)을 설계하는 데 도움이 된다.
- diagnosis-informed data preparation(진단 정보 기반 데이터 준비) 때문에 깨끗한 ordinary-tabular baseline(일반 표형 기준 모델)은 아니다.

## 검증 메모
- 출처: arXiv abstract.
