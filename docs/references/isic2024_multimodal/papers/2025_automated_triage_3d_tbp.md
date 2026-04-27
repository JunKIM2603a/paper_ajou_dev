# Automated Triage with 3D Total-Body Photography

- 인용 정보: Kurtansky et al., 2025, `npj Digital Medicine`, https://doi.org/10.1038/s41746-025-02070-7
- 출판 유형: peer-reviewed(동료심사 논문)
- Seed 논문 인용 관계: SLICE-3D; 이 논문도 seed paper 2에 해당
- 데이터셋: ISIC 2024 / SLICE-3D training data(학습 데이터)와 private leaderboard evaluation data(비공개 순위표 평가 데이터)
- 과제: 3D-TBP 병변 tile에서 pathology-confirmed(병리 확인) skin cancer triage(피부암 선별)
- 모달리티: tile image(타일 영상), 기본 인구통계/위치 정보, WB360 appearance metadata(외형 메타데이터), patient context(환자 맥락)
- 추론 입력: image tile + metadata(메타데이터) + patient-contextual feature(환자 맥락 특징)
- Strict-contract 호환성: 부분 호환

## 목표와 기여
- ISIC 2024 challenge(대회) 결과를 요약하고 자동 atypical lesion triage(비전형 병변 선별)의 임상적 타당성을 평가한다.
- tile image, demographic metadata(인구통계 메타데이터), WB360 appearance metadata, patient-contextual feature에 대한 ablation(구성요소 제거 비교)을 제공한다.
- multimodal 정보와 patient-contextual 정보가 triage 성능을 실질적으로 개선함을 보인다.

## 불균형 처리
- 낮은 false-negative(거짓 음성) 운영을 강조하기 위해 80% TPR 이상 구간의 pAUC를 primary leaderboard metric(주요 순위표 지표)으로 사용한다.
- sensitivity threshold(민감도 임계값)에서 top-15 sensitivity와 NNT 같은 patient-level triage style metric(환자 단위 선별 방식 지표)을 보고한다.
- 극단적인 rare-positive setting(희귀 양성 설정)이 핵심 한계로 남아 있으며, leaderboard probing/overfitting(순위표 탐색/과적합)을 명시적으로 논의한다.

## 모델
- Tabular: metadata와 patient-contextual feature를 사용하는 gradient boosting tree(그래디언트 부스팅 트리) 모델.
- Image: Kaggle 우승 솔루션의 독립 EVA / EdgeNeXt 계열 image 모델.
- Fusion: late fusion(후기 융합); image model probability estimate(확률 추정값)와 metadata feature를 GBT 모델에 입력한다.

## 지표와 결과
- 지표: pAUC>80%TPR, full ROC-AUC, SE top-15, sensitivity threshold별 NNT.
- 최고 보고 결과: private leaderboard(비공개 순위표)에서 skin cancer pAUC 0.1726, AUC 0.9668.
- Melanoma-specific 결과: pAUC 0.1757, AUC 0.9704, SE top-15 0.7908.
- 검증/테스트 protocol(절차): Kaggle hidden public/private leaderboard(숨겨진 공개/비공개 순위표); private evaluation set에서 ablation.
- Threshold(임계값) 선택: challenge metric은 threshold-free(임계값 불필요); NNT는 고정 sensitivity threshold에서 평가.

## 한계
- Patient-contextual feature는 같은 환자의 여러 병변을 필요로 한다.
- WB360 appearance metadata는 proprietary tooling(독점 도구)에서 나오며 외부에서 항상 사용할 수 있는 것은 아니다.
- Public leaderboard overfitting(공개 순위표 과적합)을 인정하므로 private score(비공개 점수)가 더 신뢰할 만하다.

## 우리 연구와의 관련성
- ISIC 2024 metric framing(지표 구성 방식)과 multimodal feature-class ablation(멀티모달 특징군 제거 비교)에 대한 가장 강한 reference(참고 논문)이다.
- pAUC@TPR>=0.80 아래에서 image-only, tabular-only, multimodal baseline을 비교해야 한다는 필요성을 뒷받침한다.
- patient context와 proprietary feature(독점 특징)를 사용할 수 없을 수 있으므로 순수한 single-lesion strict baseline(단일 병변 엄격 기준 모델)은 아니다.

## 검증 메모
- 출처: Nature article의 pAUC metric, private score, ablation, feature class 관련 내용.
