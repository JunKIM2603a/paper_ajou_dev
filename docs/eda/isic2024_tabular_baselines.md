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

## Split Protocol

기본 실행은 locked split artifacts를 사용한다.

```text
data/splits/isic2024_train_validation_test_split_seed42.csv
data/splits/isic2024_train_validation_5fold_seed42.csv
```

이 파일이 없으면 먼저 strict input export를 실행한다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

Runner는 `cv_validation_fold` 하나를 validation fold로 쓰고, 같은 `train_validation_data` 안의 나머지 rows를 train으로 쓴다. `test_data`는 locked internal holdout이다.

## Threshold Protocol

F1, precision, recall, balanced accuracy는 validation probabilities에서 선택한 threshold를 사용한다.

```text
threshold_source = validation_f1
```

AUC, pAUC, Average Precision은 threshold-independent metric으로 probabilities에서 계산한다. Trial selection은 validation metrics만 사용하며 test metrics로 fallback하지 않는다.

## Commands

Smoke run:

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --max-train-rows 2000 \
  --max-val-rows 1000 \
  --max-test-rows 1000
```

Paper-facing run은 row cap을 사용하지 않는다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --models logistic_regression svm mlp xgboost catboost lightgbm ft_transformer \
  --feature-sets strict_main_input
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
holdout_split_csv
cv_split_csv
cv_fold
threshold_source
selected_threshold
train/val/test metrics
numeric_columns
categorical_columns
```

Generated outputs belong under:

```text
experiments/outputs/tabular_baselines/
experiments/logs/
```

Large outputs and MLflow logs are not tracked in Git.
