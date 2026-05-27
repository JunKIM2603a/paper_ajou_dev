# 2026-04-02 주간 진행 보고서

## 1. 이번 주 핵심 요약

- `ISIC2024 Challenge Benchmark Workspace`를 기준으로 `Tabular EDA -> Tabular Baselines -> Image Baselines` 의 3단계로 진행 중
- `Tabular EDA`: 완료
- `Tabular Baselines`: 완료
- `Image Baselines`: 부분 완료
- 활용 metadata: `train-metadata.csv`(401,059 sample)만 사용. `test-metadata.csv`는 3개 샘플만 있고 target 라벨 없음.

| 단계 | 현재 상태 | 산출물 | 현재 해석 |
| --- | --- | --- | --- | 
| Tabular EDA | 완료 | EDA 리포트, 결측 요약, feature set 추천 파일 생성 완료 | leakage 가능 컬럼과 현실형 feature set을 구분할 근거 확보 |
| Tabular Baselines | 완료 | 5개 모델 x 3개 feature set 실행 및 MLflow 리포트 생성 완료 | `strict`를 메인 비교표로 사용 가능 |
| Image Baselines | 부분 완료 | 13개 설정 폴더 준비, 7개 모델에서 총 22개 summary 확보 | 통합 리더보드는 아직 없고, 부분 결과 해석 단계 |

## 2. 단계별 진행 현황

### 2.1 Tabular EDA

- `train-metadata.csv` 데이터 개요
  - 전체 샘플 수: `401,059`
  - 양성 클래스 수: `393` (`0.098%`)
  - 음성 클래스 수: `400,666` (`99.902%`)
  - 라벨, 이미지, 보조 정보는 `isic_id` 기준으로 연결됨
  - 문제 유형은 극단적 클래스 불균형 이진 분류
  - 따라서 정확도보다 `average precision`, `AUC`, `balanced accuracy`, `recall` 중심으로 해석하는 것이 적절함
  - split: `train / val / internal test`로 나누며, 이때 같은 환자가 서로 다른 split에 섞이지 않도록 `split_group_id = patient_id`를 사용함

#### EDA의 핵심 목적 

> 데이터 분포 확인 및 baseline 비교를 위한 `strict / relaxed / oracle` 기준을 설계

| 설정 | 의미 |
| --- | --- |
| `strict` | 현실형 메인 baseline. 기본 임상 정보와 결측이 적은 TBP 수치형 중심 |
| `relaxed` | 보조 신호를 허용했을 때 얼마나 성능이 올라가는지 보여주는 보조 비교용 세트 |
| `oracle` | 공정한 입력이라고 보기 어려운 진단/후속기록 컬럼까지 일부러 넣었을 때 성능이 어디까지 올라가는지 보는 상한선 확인용 세트 |

#### 추가 자료

- 프로젝트 전체 개요와 실행 경로: [README](../../../../README.md)
- 데이터/모델 흐름 다이어그램: 이전 링크 `program_diagram.md`는 `docs/date/` 재배치 후 현재 저장소에 없음
- 상세 EDA 보고서 생성 경로: `experiments/evidence/eda/isic_2024/eda_report.md`, `experiments/evidence/eda/isic_2024/isic2024_presentation_only_eda.ipynb` (자동생성 산출물, 현재 저장소에는 미포함)
- feature set 추천 파일 생성 경로: `experiments/evidence/eda/isic_2024/feature_sets_recommended.json` (자동생성 산출물, 현재 저장소에는 미포함)

### 2.2 Tabular Baselines

- 모델
  1. `logistic_regression`
  2. `svm`
  3. `mlp`
  4. `xgboost`
  5. `catboost`

- feature set
  1. `strict` (trial_001)
  2. `relaxed` (trial_002)
  3. `oracle` (trial_003)

대표 결과는 아래와 같다.

| feature set | 대표 모델 | test average precision | test balanced accuracy | test AUC |
| --- | --- | --- | --- | --- |
| strict | CatBoost | 0.0978 | 0.7603 | 0.9363 |
| relaxed | CatBoost | 0.3063 | 0.8839 | 0.9939 |
| oracle | XGBoost | 1.0000 | 1.0000 | 1.0000 |

관련 기존 자료는 다음과 같다.

- Tabular 부모 run 요약 CSV 생성 경로: `experiments/tables/mlflow_leaderboard.csv` (자동생성 산출물, 현재 저장소에는 미포함)
- Tabular 전체 비교용 HTML 리포트 생성 경로: `experiments/tables/mlflow_report.html` (자동생성 산출물, 현재 저장소에는 미포함)
- feature set별 개별 trial 결과 폴더 생성 경로: `experiments/outputs/tabular_baselines` (자동생성 산출물, 현재 저장소에는 미포함)

### 2.3 Image Baselines

`Image Baselines` 단계는 아직 부분 완료 상태다. 현재 설정 폴더는 총 13개이며, 이 중 실제 `summary.json`이 확인된 모델은 7개다.

- 결과 확인된 모델: `ResNet-50`, `DenseNet-121`, `EfficientNet-B0`, `DeiT-S`, `DINOv2 ViT-S`, `CheXzero`, `BioMedCLIP`
- 아직 전체 실행이 남아 있는 설정: `EyePACS`, `HAM10000`, `MedCLIP`, `RETFound`, `TorchXRayVision`, `ViT-B_16`
- 현재 확보된 메인 결과 summary는 총 `22`개 trial이다.

현재까지 확보된 이미지 결과 중 가장 좋은 대표값은 `BioMedCLIP_trial_001`이며, `test average precision = 0.0450`, `test AUC = 0.9073`이다. 다만 이미지 쪽은 아직 전체 모델이 동일 조건으로 정리된 통합 CSV/HTML 리더보드가 없으므로, 현재 보고에서는 "파이프라인 검증 및 부분 실험 완료" 수준으로 설명하는 것이 적절하다.

추가로 `MedCLIP`은 현재 기본 환경에서 재현 문제가 있어 별도 conda 환경에서 재실행하는 방안이 이미 정리되어 있다. 따라서 image 단계는 단순히 "모델을 더 돌리면 끝"이 아니라, 일부 모델은 환경 분리 전략까지 포함해 관리할 필요가 있다.

관련 기존 자료는 다음과 같다.

- 이미지 baseline 설정 폴더: [experiments/configs/image_baselines](../../../../experiments/configs/image_baselines)
- 이미지 파이프라인 검증용 smoke 리포트 생성 경로: `experiments/outputs/image_smoke/mlflow_report.html` (자동생성 산출물, 현재 저장소에는 미포함)
- 대표 이미지 결과 예시 생성 경로: `experiments/outputs/BioMedCLIP`, `experiments/outputs/ResNet-50` (자동생성 산출물, 현재 저장소에는 미포함)
- `MedCLIP` 환경 주의사항: [README](../../../../README.md)

## 3. 향후 진행 방향
1. 가중치 초기화: fine-turning Best 모델에 load checkpoint
2. 도메인 사전학습: ISIC2024 image 23만장으로 MAE(Masked-Autoencoder) 실행
  - 목적: ViT가 "피부암 이미지"의 특성을 이해. 병변의 경계, 색상 변화 등 학습
2. Metadata  임베딩 모듈 설계 
  - 목적: ViT 의 patch token 과 함께 사용하려는 목적의 metadata를 meta token 형태로 변환
  - 구조: 정형 데이터를 ViT의 임베딩 차원(예: 768)과 똑같은 크기의 벡터(토큰)로 변환하는 MLP 레이어를 설계
    - 정형 데이터: 범주형/수치형 컬럼을 각각 인코딩한 후, 상관관계 분석하여 group vector 또는 group score 얻고, group 을 묶어 하나의 일반식으로 결합하여 하나의 통합 백터 생성 
  - Output: d_model 차원의 meta token 생성
3. 멀티모달 통합 및 웜업 학습 (Warm-up)
  - 작업: MAE로 학습된 ViT 백본은 얼려두고(Freeze), 새로 만든 메타데이터 모듈과 마지막 분류기(Classifier)만 먼저 학습
  - 이유: 갑자기 전체를 학습하면 잘 훈련된 ViT의 가중치가 무작위 상태인 메타데이터 모듈 때문에 망가질 수 있음
4. 전체 미세 조정 (End-to-End Fine-tuning)
  - 작업: ViT 백본의 락(Freeze)을 풀고, [이미지 토큰 + 메타데이터 토큰] 전체를 동시에 학습
  - 테스트 데이터에 없는 컬럼을 처리하기 위한 'Missing Modality Dropout' (일부 메타데이터 토큰을 랜덤하게 0으로 만듦) 기법을 적용하여 강건한 모델 구축
