# ISIC2024 Challenge Benchmark Workspace

이 저장소는 `dataset/isic-2024-challenge` 기준으로 tabular / image baseline을 관리합니다.
기본 명령은 공공 서버 기준 `paper_ajou_dev` conda 환경에서 실행합니다.
단, `MedCLIP`은 별도 conda 환경에서만 재검토합니다.

## 기본 경로

- dataset: `dataset/isic-2024-challenge`
- source package: `src/isic2024_benchmark`
- image configs: `src/image_baselines`
- orchestration script: `src/isic2024_benchmark/run_all_models.py`
- tabular EDA: `artifacts/eda/isic2024`
- tabular runs: `artifacts/tabular_runs`
- tabular leaderboard: `artifacts/tabular/mlflow_leaderboard.csv`
- image runs: `artifacts/<model_name>/...`
- image leaderboard: `artifacts/image_mlflow_leaderboard.csv`
- image HTML report: `artifacts/image_mlflow_report.html`
- MLflow tracking DB: `mlflow.db` (권장)
- MLflow legacy FileStore artifacts: `mlruns`

## 환경 확인

```bash
conda run -n paper_ajou_dev python --version
```

필요 패키지 설치:

```bash
conda run -n paper_ajou_dev python -m pip install -r requirements.txt
```

현재 코드 레이아웃은 `src` 기반이다.
`python -m isic2024_benchmark...` 형태의 명령은 아래처럼 `PYTHONPATH=./src`를 함께 준다.

권장 MLflow tracking URI:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

기존 `mlruns` FileStore를 SQLite로 한 번만 마이그레이션:

```bash
conda run -n paper_ajou_dev mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db --progress
```

코드 기본값은 다음 우선순위를 따릅니다.

- `MLFLOW_TRACKING_URI`가 설정돼 있으면 그 값을 사용
- `./mlflow.db`가 있으면 `sqlite:///mlflow.db` 사용
- 그 외에는 legacy `file:./mlruns` 사용

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

현재 `Tabular EDA`의 목표는 Kaggle 스타일의 시각화 나열 자체보다, 아래 질문에 답하는 것이다.

- 어떤 컬럼이 메인 baseline에 들어가도 되는가
- 어떤 컬럼이 편향 또는 leakage 위험 때문에 분리되어야 하는가
- `strict / relaxed / oracle`를 어떤 근거로 설계해야 하는가

즉, 이 저장소에서 EDA는 "예쁜 탐색"보다 "실험 근거 설계"에 더 가깝다.

수정 영향 범위는 아래처럼 구분한다.

- 문서/표현만 보강:
  `docs/isic2024_eda_report_template.md`, `src/isic2024_benchmark/render_isic2024_eda_report.py`, `README.md`
  이 경우 `feature_sets_recommended.json`과 기존 tabular/image 결과는 바뀌지 않는다.
- EDA 산출물 자체를 추가 생성:
  `src/isic2024_benchmark/isic2024_tabular_eda.py`
  이 경우 EDA를 다시 실행해야 하지만, feature set 규칙을 안 바꾸면 baseline 결과는 유지할 수 있다.
- feature set 규칙 변경:
  `src/isic2024_benchmark/tabular_feature_sets.py`
  이 경우 `feature_sets_recommended.json`, strict CSV export, tabular baseline 결과를 다시 생성해야 한다.
  image baseline은 현재 이 파일을 직접 사용하지 않으므로 직접 영향은 없다.

EDA 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.isic2024_tabular_eda --dataset-root ./dataset/isic-2024-challenge --output-dir ./artifacts/eda/isic2024
```

feature set 추천 생성:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.define_tabular_feature_sets --eda-dir ./artifacts/eda/isic2024 --output ./artifacts/eda/isic2024/feature_sets_recommended.json
```

EDA markdown 리포트 렌더링:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.render_isic2024_eda_report --eda-dir ./artifacts/eda/isic2024 --template ./docs/isic2024_eda_report_template.md --output ./artifacts/eda/isic2024/eda_report.md
```

리포트 템플릿/문장만 수정했다면 위 렌더링만 다시 실행하면 된다.
이 경우 feature set 구성과 기존 모델 학습 결과는 바뀌지 않는다.

strict CSV export:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.export_tabular_dataset --dataset-root ./dataset/isic-2024-challenge --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --feature-set strict --output ./artifacts/tabular/isic2024_strict.csv
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
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs
```

모델 하나만 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs --models catboost
```

여러 모델만 골라서 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/feature_sets_recommended.json --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs --models logistic_regression xgboost
```

CSV leaderboard:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Tabular-Benchmark --sort-metric best_pauc_above_tpr80 --output ./artifacts/tabular/mlflow_leaderboard.csv
```

HTML leaderboard:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Tabular-Benchmark --parent-sort-metric best_pauc_above_tpr80 --child-sort-metric val_pauc_above_tpr80 --output ./artifacts/tabular/mlflow_report.html
```

생성된 HTML에는 모델별 종합 leaderboard와 `strict` / `relaxed` / `oracle` 개별 leaderboard가 함께 포함됩니다.

## Image Baselines

단일 모델 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/ResNet-50/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42
```

현재 `run_experiment`는 단일 학습 1개를 여러 GPU에 분산하는 구조는 아니다.
대신 `--trial-indices`를 사용하면 특정 hyperparameter trial만 골라서 실행할 수 있다.
이 기능은 기존 결과를 바꾸지 않으며, 이미 실행 중인 프로세스에도 영향을 주지 않는다.

특정 trial만 선택 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src CUDA_VISIBLE_DEVICES=0 python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/RETFound/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --device cuda --trial-indices 0
```

`--trial-indices`는 0-based 이다. 예를 들어 `RETFound`의 현재 trial 매핑은 아래와 같다.

- `0`: `learning_rate=5e-05`, `weight_decay=0.0001`
- `1`: `learning_rate=5e-05`, `weight_decay=0.001`
- `2`: `learning_rate=1e-04`, `weight_decay=0.0001`
- `3`: `learning_rate=1e-04`, `weight_decay=0.001`

RETFou​nd trial을 2개 GPU에 나눠 실행하려면 터미널을 2개 열고 각각 아래처럼 실행한다.

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src CUDA_VISIBLE_DEVICES=0 python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/RETFound/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --device cuda --trial-indices 0
conda run -n paper_ajou_dev env PYTHONPATH=./src CUDA_VISIBLE_DEVICES=1 python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/RETFound/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --device cuda --trial-indices 1
```

위 방식은 `trial`을 서로 다른 GPU에 나눠 실행하는 것이지, 하나의 trial이 GPU 2개를 동시에 사용하는 것은 아니다.

2-GPU 병렬 전체 실행:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_all_models --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --devices 0 1 --exclude-models HAM10000
```

위 명령은 전체 모델 실행이 끝나면 자동으로 아래 산출물을 생성합니다.

- `artifacts/image_mlflow_leaderboard.csv`
- `artifacts/image_mlflow_report.html`

기본 primary metric은 `pauc_above_tpr80`이며, image trial selection은 `best_val_pauc_above_tpr80` 기준으로 정렬합니다. `src/image_baselines/MONET/config.json`이 추가되어 `chanwkim/monet`도 같은 파이프라인에서 full fine-tuning baseline으로 실행할 수 있습니다.
필요하면 `--models ...`, `--exclude-models ...`, `--config-root ...`로 notebook에서 만든 follow-up config 집합만 따로 실행할 수 있습니다.

CPU smoke 예시:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/ResNet-50/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts/image_smoke --experiment-name ISIC2024-Image-Smoke --device cpu --seed 42 --max-trials 1 --epochs-override 1 --max-train-samples 1024 --max-val-samples 256 --max-test-samples 256 --disable-pretrained
```

MedCLIP 실행 주의:

- 이번 benchmark 실행에서는 `MedCLIP`을 제외한다.
- 현재 `src/image_baselines/MedCLIP/config.json`은 실제 MedCLIP 가중치가 아니라 placeholder 성격이 남아 있다.
- 현재 확인된 conda 환경 버전은 `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `transformers 4.57.6` 입니다.
- 따라서 이름에 맞는 MedCLIP baseline을 재현하려면 `medclip` package 또는 별도 backend를 먼저 정리해야 합니다.

이번에는 실행하지 않는 이유:

- 다른 image baseline 다수는 현재 환경에서 바로 실행 가능하지만, `MedCLIP`만 backend와 환경 제약을 별도 정리해야 합니다.
- PyTorch 스택 업그레이드는 다른 모델 및 GPU wheel 선택까지 다시 검토해야 해서, 이번 benchmark 범위에서는 보수적으로 제외합니다.

향후 별도 환경에서 재검토할 경우:

- 기존 `paper_ajou_dev`를 바꾸지 말고, `medclip_py310` 같은 별도 conda 환경을 사용합니다.
- PyTorch 스택을 `torch 2.6.0`, `torchvision 0.21.0`, `torchaudio 2.6.0`으로 맞춘 뒤 다시 실행합니다.
- 공식 PyTorch 버전 매트릭스 기준으로 `torch 2.6.0`은 `torchvision 0.21.0`과 짝이 맞습니다.

GPU wheel 예시:

```bash
conda run -n medclip_py310 python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

CPU wheel 예시:

```bash
conda run -n medclip_py310 python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

별도 환경에서 MedCLIP 재실행:

```bash
conda run -n medclip_py310 env PYTHONPATH=./src ISIC2024_EXPECTED_CONDA_ENV=medclip_py310 python -m isic2024_benchmark.run_experiment --config ./src/image_baselines/MedCLIP/config.json --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --device cuda --seed 42
```

image smoke leaderboard:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.mlflow_report --experiment-name ISIC2024-Image-Smoke --sort-metric best_pauc_above_tpr80 --output ./artifacts/image_smoke/mlflow_leaderboard.csv
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.mlflow_html_report --experiment-name ISIC2024-Image-Smoke --parent-sort-metric best_pauc_above_tpr80 --child-sort-metric best_val_pauc_above_tpr80 --output ./artifacts/image_smoke/mlflow_report.html
```

## MLflow UI

```bash
conda run -n paper_ajou_dev mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

## 결과 초기화

주의: image 실행 중 내려받은 HF / torch cache는 `artifacts/cache` 아래에 저장되며, 아래 명령은 그 캐시까지 함께 삭제합니다.

```bash
rm -rf ./artifacts ./mlruns ./mlartifacts ./mlflow.db
mkdir -p ./artifacts
```
