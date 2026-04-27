# FusionNetX

- 인용 정보: Nguyen et al., 2025, `Journal of Computer Science and Cybernetics`, https://doi.org/10.15625/1813-9663/22005
- 출판 유형: peer-reviewed(동료심사 논문)
- Seed 논문 인용 관계: SLICE-3D / ISIC 2024 citation 확인 필요
- 데이터셋: ISIC 2024
- 과제: 3D-TBP lesion crop 기반 skin cancer detection
- 모달리티: image data와 metadata(메타데이터)
- 추론 입력: image + metadata
- Strict-contract 호환성: 호환 가능 / 확인 필요

## 목표와 기여
- ISIC 2024 skin cancer detection을 위한 multimodal framework(멀티모달 프레임워크)로 FusionNetX를 제안한다.
- CNN 및 Transformer image representation(영상 표현)을 tree-based classifier(트리 기반 분류기)가 처리한 metadata와 결합한다.
- 극단적인 class imbalance(클래스 불균형)와 patient-group generalization(환자 그룹 일반화)에서의 robustness(강건성)를 강조한다.

## 불균형 처리
- journal abstract(학술지 초록)에 따르면 advanced sampling technique(고급 표본 추출 기법)을 사용한다.
- patient-aware evaluation(환자 인식 평가)에 중요한 stratified group cross-validation(층화 그룹 교차검증)을 사용한다.
- paper-level claim(논문 수준 주장) 전에 정확한 sampling과 fold construction(폴드 구성)을 PDF에서 확인해야 한다.

## 모델
- Tabular: metadata를 사용하는 tree-based classifier.
- Image: CNN 및 Transformer 기반 feature extractor(특징 추출기).
- Fusion: image feature(영상 특징)를 metadata/tree model output(출력)과 통합; 정확한 late/stacked fusion(후기/스택형 융합) 세부 사항 확인 필요.

## 지표와 결과
- 지표: pAUC, private test score(비공개 테스트 점수).
- 최고 보고 결과: cross-validation(교차검증) pAUC 0.18380, private score(비공개 점수) 0.17295.
- 검증/테스트 protocol(절차): stratified group CV와 ISIC 2024 private test score.
- Threshold(임계값) 선택: abstract에는 명시되지 않음; 확인 필요.

## 한계
- fold grouping(폴드 그룹화), preprocessing fit scope(전처리 적합 범위), 정확한 fusion mechanism(융합 메커니즘)을 검증하려면 full text(본문) 세부 사항이 필요하다.
- Private leaderboard score(비공개 순위표 점수)는 유용하지만 내부에서 재현 가능한 patient-level protocol(환자 단위 절차)을 대체해서는 안 된다.
- Metadata feature list(메타데이터 특징 목록)에 diagnosis-derived(진단 유래) 또는 target-derived field(타깃 유래 필드)가 있는지 확인해야 한다.

## 우리 연구와의 관련성
- image + ordinary metadata(일반 메타데이터) multimodal baseline(멀티모달 기준 모델)에 매우 가까운 reference이다.
- CNN/Transformer + metadata/tree ensemble design(앙상블 설계)의 좋은 비교 지점이다.
- 보고된 patient-group CV(환자 그룹 교차검증)가 검증된다면 특히 관련성이 높다.

## 검증 메모
- 출처: journal landing page와 DOI page.
- 논문이 SLICE-3D 및/또는 ISIC 2024 triage paper를 명시적으로 인용하는지 확인할 것.
