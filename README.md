# ISIC2024 Benchmark Workspace

이 저장소는 현재 `ISIC2024`를 기준으로 정리 중인 benchmark 작업 공간입니다.  
현재까지는 `tabular EDA`, `tabular baseline + MLflow 리더보드`, `ISIC2024 image loader 전환 및 smoke 검증`까지 진행되어 있습니다.

## 현재 구조

- `isic2024_benchmark/`
- `image_baselines/<모델명>/config.json`
- `run_all_models.py`
- `dataset/ISIC2024/`
- `artifacts/eda/isic2024/`
- `artifacts/tabular/`
- `artifacts/tabular_runs/`
- `artifacts/image_smoke/`

## 환경 준비

공공 서버에서는 conda 가상환경을 먼저 사용합니다.

```bash
conda run -n paper_ajou_dev python --version
```

필요 패키지 설치:

```bash
conda run -n paper_ajou_dev python -m pip install -r requirements.txt
```

## ISIC2024 데이터 요약

- 전체 샘플 수: `401,059`
- 양성 수: `393`
- 음성 수: `400,666`
- 타깃 컬럼: `malignant`
- `GroundTruth`, `Supplement`, 이미지 파일은 `isic_id` 기준으로 매칭됨

현재 데이터는 극단적인 클래스 불균형 문제이므로 `accuracy` 단독 해석은 권장하지 않습니다.

## 목표 1. Tabular EDA

EDA 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.isic2024_tabular_eda --dataset-root ./dataset/ISIC2024 --output-dir ./artifacts/eda/isic2024
```

feature set 추천 생성:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.define_tabular_feature_sets --eda-dir ./artifacts/eda/isic2024 --output ./artifacts/eda/isic2024/feature_sets_recommended.json
```

baseline 입력 CSV 추출:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.export_tabular_dataset --dataset-root ./dataset/ISIC2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --feature-set strict --output ./artifacts/tabular/isic2024_strict.csv
```

주요 산출물:

- `artifacts/eda/isic2024/report.md`
- `artifacts/eda/isic2024/eda_report.md`
- `artifacts/eda/isic2024/dataset_overview.json`
- `artifacts/eda/isic2024/missingness_summary.csv`
- `artifacts/eda/isic2024/feature_sets_recommended.json`
- `artifacts/eda/isic2024/figures/`

그래프 포함 보고서를 다시 렌더링하려면 아래처럼 실행합니다.

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.render_isic2024_eda_report --eda-dir ./artifacts/eda/isic2024 --template ./docs/isic2024_eda_report_template.md --output ./artifacts/eda/isic2024/eda_report.md
```

## 목표 2. Tabular Baselines

tabular baseline 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/ISIC2024 --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs
```

CSV 리더보드 추출:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Tabular-Benchmark --sort-metric best_average_precision --output ./artifacts/tabular/mlflow_leaderboard.csv
```

HTML 리포트 추출:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Tabular-Benchmark --parent-sort-metric best_average_precision --child-sort-metric test_average_precision --output ./artifacts/tabular/mlflow_report.html
```

현재 구현된 baseline 모델:

- `logistic_regression`
- `svm`
- `mlp`
- `xgboost`
- `catboost`

현재 feature set:

- `strict`
- `relaxed`
- `oracle`

해석 원칙:

- `strict`: 현실형 baseline 기준
- `relaxed`: leakage 가능성이 있는 보조 신호 일부 포함
- `oracle`: 진단 계열 컬럼 포함, leakage 영향 확인용

## 현재 tabular 결과 해석

- `strict` 세트는 현실적인 baseline 비교용입니다.
- `relaxed`에서는 일부 모델 성능이 크게 상승하며, `lesion_id` 같은 컬럼의 영향 가능성을 보여줍니다.
- `oracle`에서는 `logistic_regression`, `svm`이 거의 완벽한 성능에 도달해 leakage가 매우 강하다는 점이 확인됐습니다.

권장 해석은 다음과 같습니다.

- 논문/메인 비교표: `strict`
- 보조 분석: `relaxed`
- leakage 확인용 상한선: `oracle`

## 목표 3. Image Baselines

현재 이미지 baseline 설정 폴더는 `image_baselines/`로 정리되어 있고, `ISIC2024` 기준 manifest/split 로더도 연결되어 있습니다.

리눅스 서버에서는 `config.json + python -m ...` 경로를 기본 실행 방식으로 사용합니다.

단일 모델 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_experiment --config ./image_baselines/ResNet-50/config.json --dataset-root ./dataset/ISIC2024 --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42
```

전체 모델 실행:

```bash
conda run -n paper_ajou_dev python ./run_all_models.py --dataset-root ./dataset/ISIC2024 --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42
```

공공 서버에서 smoke test를 먼저 돌리려면 아래처럼 conda 환경에서 실행합니다.

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_experiment --config image_baselines/ResNet-50/config.json --dataset-root ./dataset/ISIC2024 --output-root ./artifacts/image_smoke --experiment-name ISIC2024-Image-Smoke --device cpu --seed 42 --max-trials 1 --epochs-override 1 --max-train-samples 1024 --max-val-samples 256 --max-test-samples 256 --disable-pretrained
```

image smoke 리더보드/HTML 생성:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Image-Smoke --sort-metric best_auc_roc --output ./artifacts/image_smoke/mlflow_leaderboard.csv
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Image-Smoke --parent-sort-metric best_auc_roc --child-sort-metric test_auc_roc --output ./artifacts/image_smoke/mlflow_report.html
```

현재 상태:

- `ISIC2024` image manifest builder와 group-aware split은 구현 완료
- `run_experiment.py`는 `ISIC2024` 기준 데이터 로딩과 smoke 옵션을 지원
- `ResNet-50` 기준 smoke test에서 학습, summary 저장, MLflow CSV/HTML 생성까지 검증 완료
- 아직 남은 일은 `image_baselines` 전체 모델을 실제 baseline 조건으로 순차 실행하는 것입니다

전체 모델을 공공 서버에서 안전하게 나눠 실행하려면 아래 override를 활용할 수 있습니다.

- `--max-trials`
- `--epochs-override`
- `--max-train-samples`
- `--max-val-samples`
- `--max-test-samples`
- `--disable-pretrained`

## MLflow

MLflow UI:

```bash
conda run -n paper_ajou_dev mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

현재 주요 실험명:

- `ISIC2024-Tabular-Benchmark`
- `ISIC2024-Image-Smoke`
- 전체 이미지 baseline은 `ISIC2024-Image-Benchmark`로 분리 예정

부모 런(`tags.role = model_parent`)은 모델별 대표 결과 비교용입니다.  
자식 런(`tags.role = hyperparameter_trial`)은 feature set 및 하이퍼파라미터 trial 상세 기록용입니다.

## 체크포인트 메모

일부 이미지 모델은 추가 패키지 또는 별도 체크포인트가 필요합니다.

- `BioMedCLIP`, `CheXzero`: `open_clip_torch`
- `MedCLIP`: `transformers`
- `DeiT-S`, `DINOv2 ViT-S`, `RETFound`: `timm`
- `EyePACS`, `HAM10000`, `TorchXRayVision`: 별도 체크포인트가 있으면 `config.json`의 `checkpoint_path` 사용

## 정리 명령

주의: 아래 명령은 결과 파일을 삭제합니다.

결과 초기화:

```bash
rm -rf ./artifacts/tabular ./artifacts/tabular_runs ./artifacts/image_smoke ./artifacts/eda/isic2024/figures
```

캐시까지 포함해서 초기화:

```bash
rm -rf ./artifacts/cache ./artifacts/image_smoke ./mlruns
```
