# ISIC2024 Tabular Baselines

이 문서는 ISIC2024 strict tabular baseline 시험의 입력, 전처리, split, metric, 모델 설정, 실행 방법을 정리한다. 중심 질문은 다음이다.

```text
ordinary inference-time tabular metadata만으로 malignant signal을 얼마나 측정할 수 있는가?
```

Paper-facing tabular baseline은 `strict_main_input`을 기준으로 설명한다. `iddx_full`, diagnosis text, pathology-derived fields, oracle diagnosis, target-derived feature는 ordinary inference-time input이 아니다.

## Tabular Baseline Trial Summary

[Feature Engineering / Input]

- 입력 feature set: `strict_main_input`
- 사용 feature 수: 39개
- 입력 성격: ISIC2024 ordinary metadata only
- 제외: `iddx_full`, diagnosis/pathology/oracle/target-derived column
- numeric feature: `age_approx`, `clin_size_long_diam_mm`, 다수의 `tbp_lv_*`
- categorical feature: `sex`, `anatom_site_general`, `tbp_tile_type`, `tbp_lv_location`, `tbp_lv_location_simple`
- numeric 결측치: train fold median imputation
- numeric scaling: 일반 sklearn/GBDT/torch path는 `StandardScaler`, CatBoost path는 scaling 없음
- missing indicator: `age_approx__missing`
- categorical 결측치: `"__missing__"` category
- 일반 sklearn/GBDT/torch 계열 categorical 처리: `OneHotEncoder(handle_unknown="ignore")`
- CatBoost categorical 처리: one-hot이 아니라 native categorical feature로 전달

[학습 전략]

- split: patient-level nested CV, 5 outer x 4 inner
- 학습: `inner_train`
- model/hyperparameter/threshold 선택: `inner_validation`
- 최종 평가: `outer_test`
- patient overlap: train/validation/test 간 0이어야 함
- target: malignant binary
- primary metric: `pAUC above TPR 0.80`
- 함께 보고하는 metric: AUC, Average Precision, F1, precision, recall, balanced accuracy
- threshold: validation F1 기준 선택, `threshold_source = inner_validation_f1`
- class imbalance: row-level malignant positive 비율이 약 0.1%인 ultra-rare target
- imbalance 처리:
  - Logistic Regression / SVM CPU sklearn path: `class_weight="balanced"`
  - XGBoost / LightGBM: train fold 기준 `scale_pos_weight = negative / positive`
  - torch BCE 계열 tabular estimator: train fold 기준 `pos_weight = negative / positive`
  - CatBoost: `auto_class_weights="Balanced"`

[Baseline 설정]

| model | baseline role | current backend | key setting | imbalance handling |
|---|---|---|---|---|
| `logistic_regression` | 가장 단순한 선형 확률 sanity baseline | sklearn CPU 또는 torch GPU path | `C=1.0`, `max_iter=1000` | CPU: `class_weight="balanced"`; GPU torch: BCE `pos_weight` |
| `svm` | large-margin 선형 baseline | sklearn CPU 또는 torch GPU path | `C=1.0`, `max_iter=20000` | CPU: `class_weight="balanced"`; GPU torch: squared hinge loss |
| `mlp` | 기본 nonlinear neural tabular baseline | sklearn CPU 또는 torch GPU path | hidden layers `(64, 32)`, `alpha=1e-4`, `max_iter=50` | GPU torch: BCE `pos_weight`; CPU sklearn: no class weight |
| `xgboost` | sparse-aware scalable GBDT baseline | XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | `scale_pos_weight` |
| `lightgbm` | independent efficient GBDT robustness baseline | LightGBM CPU | `n_estimators=300`, `num_leaves=31`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | `scale_pos_weight` |
| `catboost` | categorical-aware GBDT baseline | CatBoost | `iterations=300`, `depth=6`, `learning_rate=0.05` | `auto_class_weights="Balanced"` |
| `ft_transformer` | tabular Transformer / deep tabular baseline | repo-native PyTorch | `d_token=64`, `n_blocks=2`, `n_heads=4`, `lr=1e-3`, `batch_size=2048` | BCE `pos_weight` |
| `ft_transformer_external` | external implementation compatibility candidate | optional `rtdl_revisiting_models` | same high-level role as `ft_transformer` | BCE `pos_weight` |

[현재 하지 않은 것 / 주의사항]

- 현재 strict baseline은 `MinMaxScaler` 또는 `RobustScaler`를 쓰지 않는다.
- 현재 strict baseline은 Label Encoding + Embedding 방식의 categorical pipeline을 쓰지 않는다.
- 현재 strict baseline은 row-level random split이나 plain Stratified K-Fold를 쓰지 않는다.
- `TabNet`은 현재 tabular baseline runner의 supported model이 아니다.
- LightGBM 설정은 `dart booster`나 `is_unbalance=True`가 아니라 `scale_pos_weight` 기반이다.
- `logistic_regression`, `svm`, `mlp`는 CPU/sklearn path와 GPU/torch path의 estimator 구현이 다르므로 같은 모델명이라도 backend를 함께 기록해야 한다.

## Supported And Current Suite Models

현재 tabular runner가 인식하는 supported model은 다음 8개다.

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

현재 suite config인 `experiments/configs/suites/tabular_baselines.json`은 strong tabular 비교를 위해 아래 subset을 실행 대상으로 둔다.

```text
xgboost
catboost
lightgbm
ft_transformer
ft_transformer_external
```

즉, supported model set과 current suite model subset은 의도적으로 다르다. 단일 runner나 all-model runner에서는 `logistic_regression`, `svm`, `mlp`도 명시적으로 실행할 수 있다.

## Input And Missing Value Policy

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
  numeric scaling: StandardScaler
  numeric missing indicator: age_approx__missing
  categorical: constant "__missing__" fill + one-hot encoding

catboost
  numeric: train median imputation
  numeric missing indicator: age_approx__missing
  categorical: constant "__missing__" fill
  categorical encoding: one-hot을 하지 않고 CatBoost native cat_features로 전달
```

CatBoost의 categorical 처리는 다른 모델과 의도적으로 다르다. CatBoost는 categorical feature를 native categorical 입력으로 받을 수 있으므로, categorical 결측은 `__missing__` 문자열 category로 보존하고 `cat_features`에 포함한다. Numeric 결측 정책은 다른 모델과 비교 가능하도록 train median + missing indicator로 통일한다.

## Split And Threshold Protocol

기본 실행은 patient-level Triple Stratified Nested CV artifact를 사용한다.

```text
data/splits/isic2024_official_train_nested_5x4_seed42.csv
```

이 파일이 없으면 먼저 strict input export를 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

Runner는 기본적으로 `outer_fold=0`, `inner_fold=0`을 읽는다. 선택된 `outer_fold`의 `outer_test`는 최종 평가 전용이고, 같은 outer fold의 `cv_train` 내부에서 `inner_validation` 하나를 validation fold로, 나머지를 `inner_train`으로 쓴다.

현재 tabular runner의 nested CV 실행 흐름은 다음과 같다.

```text
outer_fold k, inner_fold j 하나 선택
  outer_test = 최종 평가용 partition
  inner_train = 학습 partition
  inner_validation = 선택 partition

  여러 hyperparameter trial:
    - inner_train으로 preprocessing fit + model fit
    - inner_validation metric으로 best hyperparameter 선택

  best hyperparameter 확정 후:
    - 같은 inner_train으로 best 설정 재학습
    - inner_validation에서 threshold 선택
    - outer_test 평가
```

F1, precision, recall, balanced accuracy는 validation probabilities에서 선택한 threshold를 사용한다.

```text
threshold_source = inner_validation_f1
```

AUC, pAUC, Average Precision은 threshold-independent metric으로 probabilities에서 계산한다. Trial selection은 validation metrics만 사용하며 test metrics로 fallback하지 않는다.

현재 fold-wise baseline 결과를 만들 때는 `run_all_tabular_models --all-folds`로 nested split artifact 안의 모든 `(outer_fold, inner_fold)` 조합을 자동 실행할 수 있다. `--all-folds`는 5x4 nested split 기준으로 20개 실행을 만든다. 이 20개 결과는 inner fold 반복을 포함하므로 단순 평균하지 말고, 현재 요약 도구로 outer fold별 validation-selected 대표 실행을 정리한다.

현재 summary 단계는 full `cv_train` refit을 하지 않는다. `summarize_nested_cv_results`는 이미 끝난 20개 실행의 `summary.json`을 읽고, validation metric 기준으로 outer fold별 대표 실행 하나를 골라 `validation-selected nested summary`를 만든다.

논문 final model 확정 후에는 다음 절차가 paper-final refit으로 별도 필요하다.

```text
outer_fold k
  outer_test는 계속 최종 평가용으로 잠금
  inner CV 결과로 outer fold별 best hyperparameter 확정
  best hyperparameter로 full cv_train에서 train-only preprocessing + model refit
  outer_test에서 한 번 평가
```

즉, 현재 baseline summary는 모델 비교와 후보 축소용으로 쓰고, full `cv_train` refit 여부는 최종 논문 표에서 별도로 기록한다. Outer test metric이 높은 run을 골라내면 paper-valid selection이 아니다.

## Runtime And Backend Policy

Tabular baseline은 기본적으로 GPU 우선 `auto` 정책으로 실행한다. CUDA가 사용 가능하고 tensor allocation이 성공하면 GPU를 쓰고, GPU가 없거나 초기화에 실패하면 CPU로 자동 fallback한다. all-model runner에서 CPU를 강제하려면 `--device-policy cpu`, 단일 runner에서 CPU를 강제하려면 `--device cpu`를 사용한다.

GPU 사용 전에는 `paper` 환경에서 PyTorch CUDA가 실제 GPU를 볼 수 있는지 확인한다.

```bash
conda run -n paper python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

`torch.cuda.is_available()`가 `True`이고 `torch.cuda.device_count()`가 1 이상이면 GPU 실행이 가능하다. 그렇지 않으면 runner가 CPU로 fallback하고 `requested_device`, `resolved_device`, `effective_device`, `device_fallback_reason`을 summary/log에 남긴다. Driver 535 / CUDA 12.x 환경에서는 CUDA 13 wheel이 맞지 않으므로, 현재 확인한 기준은 다음과 같다.

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
  --device cpu / --device-policy cpu: sklearn estimator
  auto/cuda GPU 사용 시: repo-native torch estimator
```

따라서 paper-facing tabular baseline 비교에서 `logistic_regression`, `svm`, `mlp`의 CPU 결과와 GPU 결과는 같은 estimator 구현으로 해석하지 않는다. GPU 실행은 주로 `xgboost`, `catboost`, `ft_transformer`, `ft_transformer_external`에 사용한다. `lightgbm`은 WSL/CUDA 환경에서 CPU baseline으로 기록한다.

실행 로그는 `[YYYY-MM-DD HH:MM:SS]` prefix로 preflight, 모델별 subprocess, report 생성의 시작/종료와 duration을 남긴다. 개별 `run_tabular_baseline` subprocess 내부에서는 data/protocol load, trial, final_test 시간이 출력되고, 각 `summary.json`의 `timing_seconds`에 `prepare_splits_seconds`, `build_estimator_seconds`, `fit_seconds`, `select_threshold_seconds`, `evaluate_train_seconds`, `evaluate_val_seconds`, `evaluate_test_seconds`가 저장된다.

## Commands

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

빠른 smoke test는 paper-facing output과 섞이지 않도록 별도 output root에 쓴다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw/isic_2024_challenge \
  --models xgboost \
  --feature-sets strict_main_input \
  --devices 0 \
  --max-train-rows 2000 \
  --max-val-rows 1000 \
  --max-test-rows 1000 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports
```

GPU paper-facing candidate run은 current suite model subset을 기준으로 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw/isic_2024_challenge \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0
```

All-model runner는 supported model 전체를 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression svm mlp xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input
```

All nested folds:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --all-folds
```

Nested CV 결과 요약:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.summarize_nested_cv_results \
  --family tabular_baselines \
  --run-group-id <run_group_id>
```

Unit tests:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_tabular_baseline_protocol.py
```

## Output Evidence

Each summary records:

```text
model_name
hyperparameters
requested_device
resolved_device
effective_device
device_fallback_reason
estimator_backend
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

`run_all_tabular_models --all-folds`로 생성된 큰 산출물은 `experiments/outputs/` 아래에 두고 Git에 올리지 않는다. Git에는 outer fold별 선택 결과와 metric 요약만 작게 산출해서 올린다.

기본 nested CV summary 출력 위치는 다음과 같다.

```text
experiments/tables/tabular_baselines/<run_group_id>/nested_cv/
```

생성되는 Git-friendly 산출물은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `nested_cv_all_candidates.csv` | 모든 `(outer_fold, inner_fold, model)` 후보의 validation/test 요약 |
| `nested_cv_outer_selection.csv` | outer fold별 validation-selected 대표 실행 1개 |
| `nested_cv_metric_summary.csv` | 선택된 outer fold들의 test metric mean/std/min/max |
| `nested_cv_summary.md` | 발표/공유용 짧은 Markdown 요약 |
| `nested_cv_summary.json` | 위 내용을 재사용하기 위한 machine-readable manifest |
