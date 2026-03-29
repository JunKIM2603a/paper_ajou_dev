# ISIC2024 Challenge Benchmark Workspace

이 저장소는 `dataset/isic-2024-challenge` 기준으로 tabular / image baseline을 관리합니다.
모든 명령은 공공 서버 기준 `paper_ajou_dev` conda 환경에서 실행합니다.

## 기본 경로

- dataset: `dataset/isic-2024-challenge`
- tabular EDA: `artifacts/eda/isic2024`
- tabular runs: `artifacts/tabular_runs`
- tabular leaderboard: `artifacts/tabular/mlflow_leaderboard.csv`
- image runs: `artifacts/<model_name>/...`
- image leaderboard: `artifacts/image_mlflow_leaderboard.csv`
- image HTML report: `artifacts/image_mlflow_report.html`
- MLflow tracking: `mlruns`

## 환경 확인

```bash
conda run -n paper_ajou_dev python --version
```

필요 패키지 설치:

```bash
conda run -n paper_ajou_dev python -m pip install -r requirements.txt
```

## 데이터 개요

- metadata source: `dataset/isic-2024-challenge/train-metadata.csv`
- image source: `dataset/isic-2024-challenge/train-image/image/*.jpg`
- rows: `401,059`
- positives: `393`
- negatives: `400,666`
- target column: `target`
- split policy: `patient_id -> lesion_id -> isic_id`

동일 환자의 샘플이 매우 많기 때문에 tabular / image 모두 환자 단위 split을 사용합니다.

## Tabular EDA

EDA 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.isic2024_tabular_eda --dataset-root ./dataset/isic-2024-challenge --output-dir ./artifacts/eda/isic2024
```

feature set 추천 생성:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.define_tabular_feature_sets --eda-dir ./artifacts/eda/isic2024 --output ./artifacts/eda/isic2024/feature_sets_recommended.json
```

EDA markdown 리포트 렌더링:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.render_isic2024_eda_report --eda-dir ./artifacts/eda/isic2024 --template ./docs/isic2024_eda_report_template.md --output ./artifacts/eda/isic2024/eda_report.md
```

strict CSV export:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.export_tabular_dataset --dataset-root ./dataset/isic-2024-challenge --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --feature-set strict --output ./artifacts/tabular/isic2024_strict.csv
```

## Tabular Baselines

실행 모델:

- `logistic_regression`
- `svm`
- `mlp`
- `xgboost`
- `catboost`

feature set:

- `strict`
- `relaxed`
- `oracle`

baseline 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs
```

CSV leaderboard:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Tabular-Benchmark --sort-metric best_average_precision --output ./artifacts/tabular/mlflow_leaderboard.csv
```

HTML leaderboard:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Tabular-Benchmark --parent-sort-metric best_average_precision --child-sort-metric test_average_precision --output ./artifacts/tabular/mlflow_report.html
```

## Image Baselines

단일 모델 실행:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_experiment --config ./image_baselines/ResNet-50/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42
```

2-GPU 병렬 전체 실행:

```bash
conda run -n paper_ajou_dev python ./run_all_models.py --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --devices 0 1
```

위 명령은 전체 모델 실행이 끝나면 자동으로 아래 산출물을 생성합니다.

- `artifacts/image_mlflow_leaderboard.csv`
- `artifacts/image_mlflow_report.html`

CPU smoke 예시:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.run_experiment --config image_baselines/ResNet-50/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts/image_smoke --experiment-name ISIC2024-Image-Smoke --device cpu --seed 42 --max-trials 1 --epochs-override 1 --max-train-samples 1024 --max-val-samples 256 --max-test-samples 256 --disable-pretrained
```

image smoke leaderboard:

```bash
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Image-Smoke --sort-metric best_auc_roc --output ./artifacts/image_smoke/mlflow_leaderboard.csv
conda run -n paper_ajou_dev python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Image-Smoke --parent-sort-metric best_auc_roc --child-sort-metric test_auc_roc --output ./artifacts/image_smoke/mlflow_report.html
```

## MLflow UI

```bash
conda run -n paper_ajou_dev mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

## 결과 초기화

주의: image 실행 중 내려받은 HF / torch cache는 `artifacts/cache` 아래에 저장되며, 아래 명령은 그 캐시까지 함께 삭제합니다.

```bash
rm -rf ./artifacts ./mlruns
mkdir -p ./artifacts
```
