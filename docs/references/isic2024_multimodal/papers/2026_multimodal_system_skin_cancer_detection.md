# Multimodal System for Skin Cancer Detection

- 인용 정보: Sydorskyi et al., 2026, arXiv:2601.14822, https://doi.org/10.48550/arXiv.2601.14822
- 출판 유형: preprint(사전공개 논문); `System research and information technologies` 게재 승인
- Seed 논문 인용 관계: SLICE-3D citation 가능성 있음; 정확한 cited-by relation 확인 필요
- 데이터셋: ISIC 2024, ISIC Archive, generated data(생성 데이터)
- 과제: conventional photo image(일반 사진 영상)를 사용한 melanoma / malignant lesion detection(악성 병변 탐지)
- 모달리티: image와 tabular metadata(표형 메타데이터)
- 추론 입력: image + tabular metadata; metadata가 없는 case(사례)도 지원
- Strict-contract 호환성: 호환 가능 / 부분 호환

## 목표와 기여
- conventional, non-dermoscopic photo(일반 비더모스코피 사진)를 사용하는 접근 가능한 multimodal melanoma detection system(멀티모달 흑색종 탐지 시스템)을 구축한다.
- metadata-aware(메타데이터 사용) 및 metadata-missing setting(메타데이터 누락 설정)을 위한 one-stage, two-stage, three-stage pipeline(1/2/3단계 파이프라인)을 연구한다.
- vision architecture(비전 구조), boosting algorithm(부스팅 알고리즘), loss function(손실 함수)에 대한 ablation(구성요소 제거 비교)을 제공한다.

## 불균형 처리
- malignant example(악성 예시)을 늘리기 위해 external ISIC Archive data(외부 ISIC Archive 데이터)와 generated data를 사용한다.
- 극단적인 rare-positive condition(희귀 양성 조건)에서 loss function과 multi-stage refinement(다단계 정제)를 평가한다.
- 사용 가능한 source snippet(출처 발췌)에 따르면 ISIC Archive data에서 ISIC 2024 patient(환자)를 제외한다.

## 모델
- Tabular: XGBoost와 LightGBM variant(변형).
- Image: Multi-Modal ConvNeXt, EdgeNeXt, 기타 최신 vision architecture.
- Fusion: multimodal neural network(멀티모달 신경망)와 two-stage/three-stage boosted refinement.

## 지표와 결과
- 지표: Partial ROC AUC, ROC AUC, Top-15 retrieval sensitivity(검색 민감도).
- 최고 보고 결과: peak(최고) pAUC 0.18068, top-15 retrieval sensitivity 0.78371.
- 검증/테스트 protocol(절차): secondary review snippet에 따르면 5-fold CV(5폴드 교차검증)와 Kaggle public/private benchmark(공개/비공개 기준 평가).
- Threshold(임계값) 선택: 명확히 명시되지 않음; 확인 필요.

## 한계
- preprint 상태이므로 강한 paper claim(논문 주장) 전에 PDF에서 세부 사항을 확인해야 한다.
- External/generated data(외부/생성 데이터) 때문에 strict ISIC-only baseline(엄격한 ISIC-only 기준 모델)과 직접 비교하기 어렵다.
- Leaderboard benchmark(순위표 기준 평가)는 유용하지만 유일한 evidence(근거)가 되어서는 안 된다.

## 우리 연구와의 관련성
- multimodal ConvNeXt/boosting과 metadata-missing robustness(메타데이터 누락 강건성)에 대한 강한 engineering reference(공학적 참고 논문)이다.
- imbalance/loss/threshold ablation planning(불균형/손실/임계값 제거 비교 계획)에 유용하다.
- external/generative augmentation(외부/생성 증강)은 strict baseline comparison(엄격 기준 모델 비교)과 분리해야 한다.

## 검증 메모
- 출처: arXiv abstract와 secondary review/source snippet.
