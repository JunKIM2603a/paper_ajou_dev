# 2026-04-01 주간 진행 보고서

## 1. 이번 주 핵심 요약

본 프로젝트는 `ISIC2024 Challenge Benchmark Workspace`를 기준으로 `Tabular EDA -> Tabular Baselines -> Image Baselines`의 3단계로 진행 중이다. 현재 기준으로 `Tabular EDA`와 `Tabular Baselines`는 구현과 주요 산출물 정리가 완료되었고, `Image Baselines`는 실행 파이프라인과 일부 모델 결과까지 확보한 상태이지만 아직 전체 모델 통합 비교표는 완성되지 않았다.

즉, 이번 보고의 핵심은 다음과 같다.

- 1단계 `Tabular EDA` 완료
- 2단계 `Tabular Baselines` 완료
- 3단계 `Image Baselines` 부분 완료
- 현재 교수님께 설명할 때는 `strict baseline`을 메인 비교 기준으로 삼는 것이 가장 타당함

| 단계 | 현재 상태 | 핵심 근거 | 현재 해석 |
| --- | --- | --- | --- |
| Tabular EDA | 완료 | EDA 리포트, 결측 요약, feature set 추천 파일 생성 완료 | leakage 가능 컬럼과 현실형 feature set을 구분할 근거 확보 |
| Tabular Baselines | 완료 | 5개 모델 x 3개 feature set 실행 및 MLflow 리포트 생성 완료 | `strict`를 메인 비교표로 사용 가능 |
| Image Baselines | 부분 완료 | 13개 설정 폴더 준비, 7개 모델에서 총 22개 summary 확보 | 통합 리더보드는 아직 없고, 부분 결과 해석 단계 |

## 2. 단계별 진행 현황

### 2.1 Tabular EDA

`Tabular EDA` 단계는 완료되었다. 현재 데이터는 총 `401,059`건이며, 이 중 양성은 `393`건으로 양성 비율이 약 `0.000980`이다. 따라서 정확도보다 `average precision`, `AUC`, `balanced accuracy`, `recall` 중심으로 해석해야 한다.

이번 EDA에서 확인한 핵심 포인트는 다음과 같다.

- `iddx_1`, `iddx_full`은 타깃과 거의 직접 연결되어 `oracle` 성격이 강하다.
- `mel_thick_mm`는 사실상 양성에만 값이 존재하여 일반 baseline feature로 사용하기 어렵다.
- `lesion_id`도 성능 상승을 유도할 수 있어 `strict`와 분리해서 다뤄야 한다.
- `tbp_lv_dnn_lesion_confidence`는 상대적으로 안정적인 핵심 수치형 신호로 해석된다.

즉, EDA 결과는 단순 탐색을 넘어서 이후 baseline 비교를 위한 `strict / relaxed / oracle` 정책을 정당화하는 역할을 한다.

활용 가능한 기존 자료는 다음과 같다.

- 프로젝트 전체 개요와 실행 경로: [README](../../../../README.md)
- 데이터/모델 흐름 다이어그램: 이전 링크 `program_diagram.md`는 `docs/date/` 재배치 후 현재 저장소에 없음
- 전환 계획 및 이전 정리 문서: 이전 링크 `isic2024_transition_plan.md`는 `docs/date/` 재배치 후 현재 저장소에 없음
- 상세 EDA 보고서 생성 경로: `experiments/evidence/eda/isic_2024/eda_report.md` (자동생성 산출물, 현재 저장소에는 미포함)
- feature set 추천 파일 생성 경로: `experiments/evidence/eda/isic_2024/feature_sets_recommended.json` (자동생성 산출물, 현재 저장소에는 미포함)

### 2.2 Tabular Baselines

`Tabular Baselines` 단계는 완료되었다. 현재 기준으로 `logistic_regression`, `svm`, `mlp`, `xgboost`, `catboost` 총 5개 모델이 실행되었고, 각 모델에 대해 `strict / relaxed / oracle` 3개 feature set 실험이 정리되어 있다.

대표 결과는 아래와 같다.

| feature set | 대표 모델 | test average precision | test balanced accuracy | test AUC |
| --- | --- | --- | --- | --- |
| strict | CatBoost | 0.0978 | 0.7603 | 0.9363 |
| relaxed | CatBoost | 0.3063 | 0.8839 | 0.9939 |
| oracle | XGBoost | 1.0000 | 1.0000 | 1.0000 |

이 결과는 다음처럼 해석하는 것이 적절하다.

- `strict`는 현실적인 입력 정보만으로 비교하는 메인 baseline이다.
- `relaxed`는 보조 신호를 허용했을 때 얼마나 성능이 올라가는지 보여주는 참고 실험이다.
- `oracle`의 거의 완벽한 성능은 진단성 컬럼이 강한 leakage를 포함한다는 증거로 해석해야 한다.

따라서 교수님께는 `strict`를 메인 표로 제시하고, `relaxed`와 `oracle`은 해석용 보조 결과로 설명하는 것이 가장 설득력 있다.

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

## 3. 현재 기준 권장 설명 방식

교수님께 설명할 때는 아래 순서를 추천한다.

1. 먼저 이 프로젝트가 `tabular`와 `image`를 함께 비교하는 benchmark 구축 작업임을 짧게 설명한다.
2. 그다음 `Tabular EDA`에서 leakage 가능 컬럼을 정리했고, 이를 바탕으로 `strict / relaxed / oracle`의 해석 프레임을 만들었다고 설명한다.
3. `Tabular Baselines`에서는 이미 비교 가능한 결과가 나왔고, 메인 비교는 `strict` 기준으로 보겠다고 말한다.
4. `Image Baselines`는 파이프라인과 일부 모델 결과까지 확보했지만, 아직 전체 모델을 동일 조건으로 끝내지 못해 현재는 중간 단계라고 설명한다.
5. 마지막으로 다음 액션을 "이미지 전체 리더보드 완성"과 "최종 비교표 통합"으로 제시한다.

이 방식의 장점은, 이미 완료된 tabular 결과를 중심 축으로 삼으면서도 image 단계의 미완료 상태를 과장하지 않고 정확하게 전달할 수 있다는 점이다.

## 4. 다음 단계 제안

현재 시점에서 가장 추천하는 다음 단계는 아래와 같다.

- `Tabular` 메인 결과 표는 `strict` 기준으로 확정하고, `relaxed / oracle`은 부록 또는 보조 해석 표로 분리한다.
- `Image Baselines`는 남은 6개 설정 중 환경 의존성이 낮은 모델부터 우선 실행해 통합 리더보드를 완성한다.
- `MedCLIP`은 기본 환경을 건드리지 말고 별도 conda 환경에서 재검토한다.
- 다음 보고서부터는 `tabular vs image`를 같은 형식의 비교표로 맞춰 보여줄 수 있도록 이미지 결과 CSV/HTML를 먼저 정리하는 것이 좋다.

## 5. 한 줄 결론

현재 프로젝트는 `Tabular` 영역에서는 비교 가능한 기준선이 이미 확보되었고, `Image` 영역은 파이프라인과 부분 실험이 끝난 상태이므로, 다음 주 핵심 목표는 `Image Baselines`의 통합 비교표를 완성하는 것이다.

## 6. 교수님 미팅용 발표 순서 메모

### 6.1 5분 발표 순서

1. 프로젝트 목표
   "이번 작업은 ISIC2024 데이터에서 `tabular baseline`과 `image baseline`을 같은 프레임으로 비교할 수 있는 benchmark 환경을 만드는 것이 목적입니다."
2. 이번 주 핵심 결론
   "현재는 `Tabular EDA`와 `Tabular Baselines`는 정리되었고, `Image Baselines`는 파이프라인과 일부 모델 결과까지 확보한 상태입니다."
3. Tabular EDA 설명
   "EDA에서 가장 중요했던 점은 데이터가 극단적으로 불균형하다는 점과, 일부 컬럼이 사실상 정답 힌트 역할을 한다는 점입니다."
   "그래서 이후 실험을 `strict / relaxed / oracle`로 나눠 해석할 수 있는 기준을 먼저 만들었습니다."
4. Tabular Baseline 설명
   "Tabular은 현재 5개 모델과 3개 feature set 실험이 끝났고, 메인 비교는 `strict` 기준으로 보는 것이 가장 타당하다고 판단했습니다."
   "현재 `strict`에서는 CatBoost가 가장 좋은 결과를 보였고, `average precision`은 `0.0978`, `AUC`는 `0.9363`입니다."
5. Image Baseline 설명
   "Image는 전체 구조와 실행 코드는 준비되었고, 현재 13개 설정 중 7개 모델에서 총 22개 trial 결과를 확보했습니다."
   "다만 아직 전체 모델을 같은 조건으로 끝내지 못해서, 현재는 중간 결과로 보는 것이 맞습니다."
6. 다음 단계 제안
   "다음 우선순위는 `Image Baselines`의 통합 리더보드를 완성하고, 이후 `tabular vs image` 비교표를 하나로 정리하는 것입니다."

### 6.2 발표 시작 멘트 예시

아래 문장으로 시작하면 전체 구조를 빠르게 잡기 좋다.

"이번 주에는 프로젝트를 세 단계로 나눠서 진행했습니다. 첫 번째는 `Tabular EDA`, 두 번째는 `Tabular Baselines`, 세 번째는 `Image Baselines`입니다. 현재는 앞의 두 단계는 정리가 끝났고, 세 번째 단계는 일부 모델 결과까지 확보한 상태입니다."

### 6.3 단계별로 꼭 말하면 좋은 한 문장

- `Tabular EDA`: "EDA의 목적은 단순 요약이 아니라, 어떤 컬럼이 현실적인 입력이고 어떤 컬럼이 leakage 가능성이 있는지 구분하는 기준을 만드는 것이었습니다."
- `Tabular Baselines`: "Tabular 결과는 이미 비교 가능한 수준까지 나왔고, 메인 성능 비교는 `strict` feature set 기준으로 보는 것이 적절합니다."
- `Image Baselines`: "Image는 파이프라인과 일부 결과는 확보했지만, 아직 전체 모델 통합 비교표가 없어서 현재는 진행 중 단계로 보고 있습니다."

### 6.4 화면에서 보여주기 좋은 순서

- 먼저 현재 문서: [2026-04-01_isic2024_progress_report.md](./2026-04-01_isic2024_progress_report.md)
- 필요하면 전체 구조 설명용 문서: 이전 링크 `program_diagram.md`는 `docs/date/` 재배치 후 현재 저장소에 없음
- Tabular 결과를 보여줄 때 생성 경로: `experiments/tables/mlflow_report.html` (자동생성 산출물, 현재 저장소에는 미포함)
- EDA 세부 설명이 필요할 때 생성 경로: `experiments/evidence/eda/isic_2024/eda_report.md` (자동생성 산출물, 현재 저장소에는 미포함)
- Image는 현재 결과 예시만 간단히 보여줄 때 생성 경로: `experiments/outputs/BioMedCLIP`, `experiments/outputs/ResNet-50` (자동생성 산출물, 현재 저장소에는 미포함)

### 6.5 교수님 질문이 나올 가능성이 높은 포인트

- 왜 `oracle` 성능이 거의 1.0인가
  "진단 결과와 매우 가까운 컬럼이 포함되어 있어서, 모델 성능이라기보다 leakage 확인용 상한선으로 해석해야 합니다."
- 왜 `strict`를 메인 결과로 보나
  "실제 사용 가능한 입력만으로 비교해야 모델 간 차이를 공정하게 볼 수 있기 때문입니다."
- 왜 accuracy보다 `average precision`을 더 보나
  "양성 비율이 `0.000980`로 매우 낮아서 accuracy는 거의 의미가 없고, 양성 탐지 성능을 더 잘 보여주는 지표가 필요합니다."
- 왜 image가 아직 완전히 안 끝났나
  "모델 수가 많고, 일부는 사전학습 가중치나 환경 의존성이 있어서 실행 조건을 더 정리해야 하기 때문입니다."

### 6.6 미팅 마지막 정리 멘트 예시

"정리하면, 현재는 `Tabular` 쪽은 비교 가능한 baseline이 확보된 상태이고, 다음 단계는 `Image` 쪽 통합 결과를 마무리해서 최종적으로 두 축을 같은 형식으로 비교할 수 있게 만드는 것입니다."
