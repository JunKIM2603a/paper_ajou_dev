# ISIC2024 Tabular Baselines

이 문서는 strict tabular baseline runner의 paper-valid protocol과 실행 방법을 정리한다.

## Supported Models

현재 tabular runner가 인식하는 모델 이름은 다음과 같다.

```text
logistic_regression
svm
mlp
xgboost
catboost
lightgbm
ft_transformer
ft_transformer_external
```

`ft_transformer`는 repo-native PyTorch implementation이다. `ft_transformer_external`은 optional `rtdl_revisiting_models` dependency가 필요하며, package가 없으면 해당 모델만 명확한 ImportError로 실패한다.

## Missing Value Policy

Strict tabular baseline의 결측치 처리는 training pipeline 안에서만 수행한다. Export 단계는 raw strict input 값을 보존하고 결측치를 채우지 않는다.

현재 strict input에서 결측이 관측된 model feature는 다음과 같다.

```text
age_approx
sex
anatom_site_general
```

`lesion_id`도 결측이 많지만 identifier이므로 모델 결측 처리 대상이 아니다. `isic_id`, `patient_id`, `lesion_id`는 split과 audit metadata로만 사용한다.

`iddx_full`, diagnosis text, pathology-derived fields, `iddx_1`-`iddx_5`, `mel_mitotic_index`, `mel_thick_mm`는 strict baseline 입력이 아니다. 이 컬럼들은 결측 처리, feature engineering, inference input에 포함하지 않는다.

모든 imputer, encoder, scaler는 fold의 train split에서만 fit한다. Validation/test는 train-fitted transform만 적용한다.

모델별 정책은 다음과 같다.

```text
logistic_regression, svm, mlp, xgboost, lightgbm, ft_transformer, ft_transformer_external
  numeric: train median imputation
  numeric missing indicator: age_approx__missing
  categorical: constant "__missing__" fill + one-hot encoding

catboost
  numeric: train median imputation
  numeric missing indicator: age_approx__missing
  categorical: constant "__missing__" fill
  categorical encoding: one-hot을 하지 않고 CatBoost native cat_features로 전달
```

CatBoost의 categorical 처리는 다른 모델과 의도적으로 다르다. CatBoost는 categorical feature를 native categorical 입력으로 받을 수 있으므로, categorical 결측은 `__missing__` 문자열 category로 보존하고 `cat_features`에 포함한다. Numeric 결측 정책은 다른 모델과 비교 가능하도록 train median + missing indicator로 통일한다.

## GPU Runtime Policy

Tabular baseline은 기본적으로 CPU에서 실행한다. GPU를 사용하려면 all-model runner의 `--devices`를 우선 사용한다. 단일 runner의 `--device cuda`는 단일 모델 디버깅용으로 둔다.

논문 운영에서는 family runner를 우선 사용한다. 이 경로는 dataset spec, run group, output/table path, MLflow filter, selection registry를 한 번에 맞춘다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_experiment_family \
  --family tabular_baselines \
  --config experiments/configs/suites/tabular_baselines.json \
  --run-group-id tabular_strict_v1_gpu0 \
  --devices 0
```

결과는 다음 위치에 저장된다.

```text
experiments/outputs/tabular_baselines/<run_group_id>/
experiments/tables/tabular_baselines/<run_group_id>/
experiments/registry/selections/best_tabular_by_run_group.json
```

Tabular family만 초기화하려면 다음처럼 실행한다. 이 명령은 image/multimodal/final output, raw data, split, registry를 삭제하지 않는다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_experiment_family \
  --family tabular_baselines \
  --run-group-id tabular_strict_v1_gpu0 \
  --reset-family-output
```

GPU 사용 전에는 `paper` 환경에서 PyTorch CUDA가 실제 GPU를 볼 수 있는지 확인한다.

```bash
conda run -n paper python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

`torch.cuda.is_available()`가 `True`이고 `torch.cuda.device_count()`가 1 이상이어야 GPU 실행이 가능하다. Driver 535 / CUDA 12.x 환경에서는 CUDA 13 wheel이 맞지 않으므로, 현재 확인한 기준은 다음과 같다.

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
lightgbm==4.6.0
```

모델별 GPU 처리 방식은 다음과 같다.

```text
xgboost
  GPU option: device="cuda", tree_method="hist"

catboost
  GPU option: task_type="GPU"
  categorical input: native cat_features 유지

lightgbm
  WSL/CUDA default: CPU backend
  note: LightGBM GPU uses OpenCL, not PyTorch/XGBoost CUDA

ft_transformer, ft_transformer_external
  PyTorch tensor/model을 CUDA device에 올려 학습
  default train/predict batch size: 2048

logistic_regression, svm, mlp
  CPU 기본값: sklearn estimator
  --device cuda 사용 시: repo-native torch estimator
```

따라서 paper-facing tabular baseline 비교에서 `logistic_regression`, `svm`, `mlp`의 CPU 결과와 GPU 결과는 같은 estimator 구현으로 해석하지 않는다. GPU 실행은 주로 `xgboost`, `catboost`, `ft_transformer`, `ft_transformer_external`에 사용한다. `lightgbm`은 WSL/CUDA 환경에서 CPU baseline으로 기록한다.

단일 GPU 권장 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0
```

여러 GPU 병렬 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0 1
```

`run_all_tabular_models --devices 0 1`은 모델별 subprocess에 `CUDA_VISIBLE_DEVICES`를 하나씩 배정하고, 각 subprocess에 `--device cuda`를 전달한다. GPU가 보이지 않거나 PyTorch CUDA 초기화가 실패하면 preflight 단계에서 중단된다.

권장 초기화는 `run_experiment_family --reset-family-output`이다. 기존 직접 runner 결과와 local MLflow history까지 모두 비우는 전체 로그 초기화가 필요할 때만 아래 명령을 쓴다.

```bash
rm -rf experiments/outputs/tabular_baselines \
       experiments/outputs/tabular_baselines_smoke \
       experiments/logs/mlruns \
       experiments/logs/mlflow.db
```

이 명령은 `data/raw`, `data/processed`, `data/splits`, `experiments/registry`를 삭제하지 않는다. Nested CV split CSV는 paper-valid 결과의 protocol 입력이므로, split을 의도적으로 재생성하는 경우가 아니면 유지한다.

실행 로그는 `[YYYY-MM-DD HH:MM:SS]` prefix로 preflight, 모델별 subprocess, report 생성의 시작/종료와 duration을 남긴다. 개별 `run_tabular_baseline` subprocess 내부에서는 data/protocol load, trial, final_test 시간이 출력되고, 각 `summary.json`의 `timing_seconds`에 `prepare_splits_seconds`, `build_estimator_seconds`, `fit_seconds`, `select_threshold_seconds`, `evaluate_train_seconds`, `evaluate_val_seconds`, `evaluate_test_seconds`가 저장된다.

`run_all_tabular_models`는 기본적으로 timestamp 기반 `run_group_id`를 생성하고, 실행 후 report를 해당 run group으로 필터링한다. 이전 MLflow run과 섞지 않고 같은 묶음을 다시 조회하려면 `--run-group-id <id>`를 명시한다.

빠른 smoke test는 paper-facing output과 섞이지 않도록 별도 output root에 쓴다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0 \
  --max-train-rows 1000 \
  --max-val-rows 500 \
  --max-test-rows 500 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports
```

## Split Protocol

기본 실행은 patient-level Triple Stratified Nested CV artifact를 사용한다.

```text
data/splits/isic2024_official_train_nested_5x4_seed42.csv
```

이 파일이 없으면 먼저 strict input export를 실행한다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

Runner는 기본적으로 `outer_fold=0`, `inner_fold=0`을 읽는다. 선택된 `outer_fold`의 `outer_test`는 최종 평가 전용이고, 같은 outer fold의 `cv_train` 내부에서 `inner_validation` 하나를 validation fold로, 나머지를 `inner_train`으로 쓴다.

논문용 fold-wise 결과는 `outer_fold=0..4`를 반복해서 만든다. 각 outer fold 내부의 model choice, hyperparameter, early stopping, threshold, calibration은 `inner_validation`에서만 수행해야 한다.

## Threshold Protocol

F1, precision, recall, balanced accuracy는 validation probabilities에서 선택한 threshold를 사용한다.

```text
threshold_source = inner_validation_f1
```

AUC, pAUC, Average Precision은 threshold-independent metric으로 probabilities에서 계산한다. Trial selection은 validation metrics만 사용하며 test metrics로 fallback하지 않는다.

## Commands

Smoke run:

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --models xgboost \
  --feature-sets strict_main_input \
  --devices 0 \
  --max-train-rows 2000 \
  --max-val-rows 1000 \
  --max-test-rows 1000 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports
```

Paper-facing run은 row cap을 사용하지 않는다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --models logistic_regression svm mlp xgboost catboost lightgbm ft_transformer \
  --feature-sets strict_main_input
```

GPU paper-facing candidate run:

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0
```

All-model runner:

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --feature-sets strict_main_input
```

Unit tests:

```bash
PYTHONPATH=./src python -m pytest tests
```

## Output Evidence

Each summary records:

```text
model_name
hyperparameters
split_source
nested_split_csv
outer_fold
inner_fold
threshold_source
selected_threshold
patient_overlap_audit
triple_balance_audit
train/val/test metrics
numeric_columns
categorical_columns
missing_value_policy
```

Generated outputs belong under:

```text
experiments/outputs/tabular_baselines/
experiments/logs/
```

Large outputs and MLflow logs are not tracked in Git.
