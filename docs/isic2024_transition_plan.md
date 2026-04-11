# ISIC2024 전환 검토 및 계획

## 현재 진행 상태

- `1st_after` -> `image_baselines` 변경 완료
- `cbis_ddsm_benchmark` -> `isic2024_benchmark` 변경 완료
- `ISIC2024` tabular EDA source of truth를 notebook으로 정리 완료
- feature set 추천(`strict_base / strict_fe / strict_main_input / relaxed`) 연결 완료
- `ISIC2024` tabular baseline + MLflow CSV/HTML 리포트 구현 완료
- `ISIC2024` image manifest builder / split / smoke 검증 완료

## ISIC2024 데이터 확인 결과

- 전체 샘플 수: `401,059`
- 양성 클래스 수: `393`
- 음성 클래스 수: `400,666`
- 라벨/이미지/보조 CSV는 `isic_id` 기준으로 매칭됨
- 문제 유형은 극단적 클래스 불균형 이진 분류

## EDA 결과 요약

- `iddx_1`, `iddx_full`은 타깃과 거의 직접 연결되는 패턴이 확인됨
- `mel_thick_mm`는 유효값이 사실상 양성에만 존재
- `lesion_id`도 `strict` 대비 `relaxed` 성능 상승 폭을 보면 강한 신호일 가능성이 있음
- `tbp_lv_dnn_lesion_confidence`는 현재 가장 안정적으로 활용 가능한 수치형 핵심 컬럼

## Feature Set 정책

### strict

- 사용 목적: 현실형 baseline
- 포함 예:
  - `attribution`
  - `copyright_license`
  - `tbp_lv_dnn_lesion_confidence`

### relaxed

- 사용 목적: 보조 신호 포함 실험
- strict + 일부 주의 컬럼 포함
- 현재는 `lesion_id`가 추가됨

### oracle

- 사용 목적: leakage 영향 확인용 상한선
- 진단 계열 컬럼 포함
  - `iddx_1`
  - `iddx_2`
  - `iddx_3`
  - `iddx_4`
  - `iddx_5`
  - `iddx_full`

## 목표 1. Tabular EDA

### 상태

- 완료

### 구현 내용

- 본선 EDA notebook: `src/eda/isic2024_eda_20260411.ipynb`
- 최종 용어 체계: `strict_raw_numeric / strict_base / strict_fe / strict_main_input / relaxed / oracle`
- 최종 tabular 입력 계약과 FE 선택 결과를 notebook에서 저장
- baseline은 notebook의 `final_inputs` 산출물을 기준으로 실행

### 산출물

- `artifacts/eda/isic2024/final_inputs/final_feature_sets_v3.json`
- `artifacts/eda/isic2024/final_inputs/final_feature_set_summary_v3.csv`
- `artifacts/eda/isic2024/thesis_plan/*.csv`
- `artifacts/eda/isic2024/final_inputs/feature_sets_recommended.json`

## 목표 2. Tabular Baseline + MLflow + HTML 리더보드

### 상태

- 기본 구현 완료
- 실제 실행 및 MLflow 리포트 생성 완료
- 일부 `relaxed/oracle` 조합은 추가 보강 여지 있음

### 구현 내용

- baseline 모델:
  - Logistic Regression
  - SVM
  - MLP
  - XGBoost
  - CatBoost
- stratified split 적용
- MLflow parent / child run 구조 적용
- CSV 리더보드 생성
- HTML 리포트 생성

### 현재 해석

- `strict_main_input`은 메인 baseline 비교용으로 적절
- `strict_base`와 `strict_fe`를 함께 비교하면 base 대비 FE 추가 이득을 분리해 볼 수 있음
- `relaxed`는 provenance/context 컬럼의 영향 확인용 보조 비교 세트

### 권장 사용 방식

1. 메인 표: `strict_main_input`
2. 분해 비교: `strict_base`, `strict_fe`
3. 보조 비교: `relaxed`

## 목표 3. Image Baseline + MLflow + HTML 리더보드

### 상태

- 부분 완료

### 현재까지 준비된 것

- 이미지 baseline 설정 폴더명 정리 완료
- `ISIC2024` image manifest builder 구현 완료
- `lesion_id` 기반 group-aware split 구현 완료
- `run_experiment.py`에서 smoke test 옵션 지원
- `ResNet-50` 1 trial smoke에서 학습, summary, MLflow CSV/HTML 생성 검증 완료
- trainer / model builder / MLflow 구조 재사용 가능

### 아직 필요한 것

1. `image_baselines` 전체 모델에 대해 실제 baseline 조건 실행
2. 모델별 사전학습 가중치 / 체크포인트 의존성 점검
3. CPU/GPU 자원에 맞는 실행 배치 전략 수립
4. 최종 이미지 실험명 `ISIC2024-Image-Benchmark`로 통합
5. 최종 image leaderboard CSV/HTML 생성

## 다음 우선순위

1. 목표 2 결과를 기준으로 메인 비교 기준을 `strict`로 확정
2. 목표 3에서 torchvision 계열 모델부터 실제 baseline 실행
3. timm / open_clip 계열의 외부 가중치 의존성 점검
4. image MLflow 리더보드 생성
5. 최종 문서/보고서 형태 정리

## 주요 리스크

- 극단적 불균형 때문에 raw accuracy는 해석 가치가 낮음
- leakage 가능 컬럼 사용 시 성능이 과대평가될 수 있음
- image 실험은 클래스 불균형 때문에 `AUC`, `average_precision`, `balanced_accuracy` 중심 해석이 필요함
- 일부 사전학습 이미지 모델은 외부 체크포인트가 더 필요할 수 있음
- 공공 서버에서는 pretrained 다운로드가 막힐 수 있으므로 `--disable-pretrained` 또는 로컬 체크포인트 전략이 필요함
