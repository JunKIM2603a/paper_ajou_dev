# Explainable Multimodal AI via 3D Imaging and Clinical Data

- 인용 정보: Wang et al., 2025, `Scientific Reports`, https://doi.org/10.1038/s41598-025-33536-z
- 출판 유형: peer-reviewed(동료심사 논문)
- Seed 논문 인용 관계: ISIC 2024 / SLICE-3D context(맥락); 정확한 seed citation 확인 필요
- 데이터셋: ISIC 2024, 1,075 patients
- 과제: six-class skin-lesion classification(6클래스 피부 병변 분류) 및 binary challenge-style benchmark(이진 대회형 기준 평가)
- 모달리티: 3D TBP image와 structured clinical/lesion feature(구조화된 임상/병변 특징)
- 추론 입력: image + 41개 clinical/lesion-specific feature(임상/병변별 특징)
- Strict-contract 호환성: 비호환 / feature-audit(특징 감사) 필요

## 목표와 기여
- skin-lesion risk prediction(피부 병변 위험 예측)을 위한 explainable multimodal AI framework(설명가능 멀티모달 AI 프레임워크)를 개발한다.
- image 기반 CNN output(출력)을 structured clinical data(구조화 임상 데이터) 및 interpretability tool(해석 도구)과 결합한다.
- SHAP과 CAM을 사용해 model behavior(모델 동작)를 clinical/lesion feature와 연결한다.
- 보고된 feature importance(특징 중요도)에는 pathology-derived(병리 유래)로 보이는 `mel_thick_mm`가 포함된다.

## 불균형 처리
- secondary/source review(2차/출처 검토)에 따르면 non-nevus class를 대상으로 targeted augmentation(표적 증강)을 사용한다.
- Binary challenge-style benchmark는 six-class classification과 별도로 보고된다.
- pFPR 표현은 pAUC@TPR>=0.80과 비교하기 전에 신중한 mapping(대응)이 필요하다.

## 모델
- Tabular: clinical-only(임상 정보 단독) XGBoost와 multinomial logistic-regression decision/scoring model(다항 로지스틱 회귀 판정/점수화 모델).
- Image: 3D TBP image로 학습한 CNN.
- Fusion: image와 structured clinical feature를 결합하는 multimodal fusion model(멀티모달 융합 모델).

## 지표와 결과
- 지표: accuracy(정확도), recall(재현율), F1, AUC, pFPR/challenge-style score(대회형 점수).
- 최고 보고 결과: recall과 F1 95% 이상, AUC 0.95 이상; challenge-style benchmark에서 pFPR 0.1734.
- 검증/테스트 protocol(절차): internal evaluation(내부 평가)과 ISIC 2024 challenge-style comparison(대회형 비교); 정확한 split(분할) 확인 필요.
- Threshold(임계값) 선택: 확인 필요.

## 한계
- Six-class diagnostic framing(6클래스 진단 구성)은 ISIC 2024 malignant binary target(악성 이진 타깃)과 동일하지 않다.
- 보고된 `mel_thick_mm`는 ordinary inference-time metadata feature(일반 추론 시점 메타데이터 특징)가 아니며, 제거하지 않으면 strict-contract 비교를 위반한다.
- pFPR은 검증 없이 repository의 pAUC 정의와 직접 교환할 수 없다.

## 우리 연구와의 관련성
- image+tabular dermatology model(피부과 모델)에서 SHAP과 CAM을 다루는 유용한 XAI reference(설명가능 AI 참고 논문)이다.
- interpretability(해석가능성)는 leakage-safe metric(누수 안전 지표)의 대체물이 아니라 secondary evidence(보조 근거)로 포함해야 함을 뒷받침한다.
- pathology-derived feature(병리 유래 특징)를 제거하고 재감사하지 않는 한 strict baseline(엄격 기준 모델)으로 직접 비교할 수 없다.

## 검증 메모
- 출처: Scientific Reports article, PubMed/PMC abstract, article discussion snippet.
