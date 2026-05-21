# ISIC2024 Reproducibility Guide

이 문서는 새 컴퓨터에서 repository를 clone한 뒤, 현재 ISIC2024 논문 실험 protocol을 같은 기준으로 재현하기 위한 실행 순서와 검증 기준을 정리한다.

기본 연구 방향은 다음으로 고정한다.

```text
lesion image + ordinary inference-time tabular metadata -> malignant probability
```

`iddx_full`은 기본 inference input이 아니다. `iddx_full` 또는 diagnosis text를 사용하는 실험은 반드시 training-only candidate 또는 analysis-only evidence로 분리한다.

## 1. Reproducibility Contract

Paper-facing 실험은 아래 조건을 모두 만족해야 한다.

| item | required rule |
|---|---|
| split | patient-level nested CV |
| preprocessing | fold-local train-only fit |
| model selection | `inner_validation` only |
| threshold selection | `inner_validation` only |
| final evaluation | `outer_test` only |
| inference input | image + ordinary tabular metadata only |
| privileged fields | no test-time dependency |
| required audit | patient overlap, fold distribution, metric source |

결과 summary에는 최소한 다음 metadata가 있어야 한다.

```text
split_protocol
nested_split_csv
outer_fold
inner_fold
threshold_source
patient_overlap_audit
triple_balance_audit
config path
seed
metric function
```

## 2. Reference Starting Point

먼저 아래 노트북을 확인한다.

```text
notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260514.ipynb
```

이 노트북은 논문 초안이 아니라, strict input contract와 split protocol을 고정하는 기준 문서다. 특히 Section 6에서 `cv_train`, `cv_test_fold`, `inner_train`, `inner_validation`, `outer_test`가 어떻게 정의되는지 확인한다.

참조 audit 노트북은 다음이다.

```text
notebooks/isic_2024/isic2024_strict_input_export_audit_20260514.ipynb
```

이 노트북은 생성된 CSV를 읽어서 strict input column, `iddx_full` exclusion, nested split role, patient overlap, fold distribution을 확인한다.

## 3. Data And Environment

Repository와 conda 환경을 준비한다.

```bash
git clone <repo-url>
cd paper_ajou_dev
conda activate paper
```

새 컴퓨터에서는 ISIC2024 raw data를 로컬에 직접 준비해야 한다. Raw data는 git으로 추적하지 않는다.

```text
data/raw/isic_2024_challenge/
```

권장 raw data 배치는 다음과 같다.

```text
data/raw/isic_2024_challenge/sample_submission.csv
data/raw/isic_2024_challenge/test-image.hdf5
data/raw/isic_2024_challenge/test-metadata.csv
data/raw/isic_2024_challenge/train-image.hdf5
data/raw/isic_2024_challenge/train-metadata.csv
```

Raw directory는 read-only로 취급한다. 생성물은 `data/processed/`, `data/splits/`, `experiments/evidence/`, `experiments/outputs/` 아래에 둔다.

아래 명령 예시는 같은 prefix를 사용한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m <module>
```

## 4. Nested Split Definition

기본 split은 patient-level Triple Stratified Nested CV다.

```text
outer folds: 5
inner folds: 4
seed: 42
artifact: data/splits/isic2024_official_train_nested_5x4_seed42.csv
```

Nested split artifact의 최소 column은 다음과 같다.

```text
isic_id
patient_id
lesion_id
outer_fold
cv_test_fold
inner_fold
split_role
```

Split role은 세 값만 허용한다.

| split_role | meaning | allowed use |
|---|---|---|
| `inner_train` | current outer train pool 안의 training partition | preprocessing fit, model fit, class weight, sampler |
| `inner_validation` | current outer train pool 안의 validation partition | model choice, hyperparameter, early stopping, threshold, calibration selection |
| `outer_test` | current outer fold의 final evaluation partition | final evaluation only |

`outer_test`는 model choice, hyperparameter search, early stopping, threshold selection, calibration fitting, preprocessing fitting에 사용할 수 없다.

핵심 용어는 다음과 같다.

| term | meaning |
|---|---|
| `official_train_pool` | ISIC2024 `train-metadata.csv` 전체 |
| `cv_test_fold` | outer fold id이며 `outer_test`와 같은 의미 |
| `cv_train` | 현재 outer test fold를 제외한 outer train pool |
| `inner_train` | `cv_train` 내부에서 다시 나눈 inner training partition |
| `inner_validation` | selection 전용 partition |
| `outer_test` | 최종 평가 전용 partition |

## 5. Reproduction Steps

### 5.1 Export Strict Inputs And Splits

Strict input, train-only `iddx_full` sidecar, nested split artifact를 생성한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw/isic_2024_challenge \
  --seed 42 \
  --outer-folds 5 \
  --inner-folds 4
```

주요 생성물은 다음과 같다.

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_official_train_nested_5x4_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

생성 직후 확인할 점:

```text
outer/inner patient overlap = 0
isic_id uniqueness preserved
iddx_full excluded from strict model input
split_role values limited to inner_train, inner_validation, outer_test
fold distribution and malignant count recorded
```

### 5.2 Run Export Audit Notebook

아래 노트북을 실행해서 generated artifacts를 검토한다.

```text
notebooks/isic_2024/isic2024_strict_input_export_audit_20260514.ipynb
```

이 단계는 paper-facing baseline 실행 전 protocol sanity check로 취급한다.

### 5.3 Run Baseline Suites With One Command

구현된 baseline family를 한 번에 시험하려면 `run_baseline_suite`를 사용한다. 기본 실행 대상은 현재 구현된 `tabular_baselines`와 `image_baselines`다.

Command preview:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_baseline_suite \
  --smoke \
  --dry-run
```

Suite preflight summary:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_baseline_suite \
  --smoke \
  --preflight-only
```

Quick smoke run:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_baseline_suite \
  --smoke \
  --devices 0
```

Full suite run:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_baseline_suite \
  --run-group-id baseline_suite_seed42 \
  --devices 0 1
```

이 명령은 내부적으로 family별 runner를 순차 호출한다.

```text
tabular_baselines -> experiments/configs/suites/tabular_baselines.json
image_baselines   -> experiments/configs/suites/image_baselines.json
```

`multimodal_baselines`는 현재 scaffold 상태이므로 기본 suite에는 포함하지 않는다. 명시적으로 확인하려면 `--families multimodal_baselines`를 지정한다.

현재 wrapper는 각 suite config의 dataset spec에 기록된 fold selection을 따른다. 논문용 전체 nested fold tabular 결과가 필요하면 아래 tabular family 명령의 `--all-folds` 흐름을 사용한다.

### 5.4 Run Tabular Baseline

Preflight로 selected fold의 split, feature set, leakage guard를 먼저 확인한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --outer-fold 0 \
  --inner-fold 0 \
  --preflight-only
```

단일 nested fold 조합을 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --outer-fold 0 \
  --inner-fold 0
```

Tabular baseline 모델 전체를 5x4 nested fold 조합으로 실행하려면 `run_all_tabular_models`를 사용한다. `--models`를 생략하면 runner의 기본 tabular model 목록 전체를 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw/isic_2024_challenge \
  --feature-sets strict_main_input \
  --all-folds
```

`--all-folds`는 20개 `(outer_fold, inner_fold)` 실행을 만든다. 위 명령은 기본 tabular model 목록 전체 x 20개 nested fold 조합을 실행한다. 특정 모델만 돌리고 싶을 때만 `--models xgboost catboost ...`처럼 subset을 명시한다.

기본 tabular model 목록:

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

각 실행은 `inner_train`으로 trial을 학습하고, `inner_validation`으로 best hyperparameter와 threshold를 선택한 뒤, 같은 `inner_train`으로 best 설정을 다시 학습하고 `outer_test`를 평가한다.

### 5.5 Summarize Nested CV Results

현재 nested summary는 `summarize_nested_cv_results`로 수행한다. 이 도구는 20개 실행의 `summary.json`을 읽고 validation metric 기준으로 outer fold별 대표 실행을 골라 validation-selected nested summary를 만든다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.summarize_nested_cv_results \
  --family tabular_baselines \
  --run-group-id <run_group_id>
```

기본 출력 위치는 다음과 같다.

```text
experiments/tables/tabular_baselines/<run_group_id>/nested_cv/
```

생성되는 Git-friendly 산출물은 다음과 같다.

| file | role |
|---|---|
| `nested_cv_all_candidates.csv` | all candidate `(outer_fold, inner_fold, model)` summaries |
| `nested_cv_outer_selection.csv` | validation-selected representative run per outer fold |
| `nested_cv_metric_summary.csv` | mean/std/min/max for selected outer-fold test metrics |
| `nested_cv_summary.md` | short human-readable summary |
| `nested_cv_summary.json` | machine-readable manifest |

Outer test metric이 높은 run을 골라 대표 실행으로 선택하면 paper-valid selection이 아니다. 대표 실행은 validation metric 기준으로만 선택한다.

Full `cv_train` refit orchestration은 아직 없다. Final paper model 확정 후 별도 단계로 추가해야 한다.

### 5.6 Run Image Baseline

Image runner도 tabular runner와 같은 nested split artifact를 읽어야 한다. Image manifest는 `isic_id`로 split artifact에 join된다.

Image baseline 모델 전체를 지정한 nested fold 조합에서 실행하려면 `run_all_image_models`를 사용한다. `--models`를 생략하면 `--config-root` 아래의 모든 image baseline config를 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_image_models \
  --dataset-root data/raw/isic_2024_challenge \
  --config-root experiments/configs/image_baselines \
  --outer-fold 0 \
  --inner-fold 0
```

위 명령은 image baseline config 전체를 같은 `(outer_fold, inner_fold)`에서 실행한다. 특정 모델만 돌리고 싶을 때만 `--models resnet50 efficientnetv2_s ...`처럼 subset을 명시한다. 현재 image orchestrator는 tabular의 `--all-folds`처럼 모든 nested fold를 자동 반복하지 않으므로, paper-facing image fold-wise 결과는 fold 조합을 명시적으로 반복해서 만든다.

단일 image model만 실행하려면 아래처럼 개별 config를 지정한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --config experiments/configs/image_baselines/resnet50/config.json \
  --outer-fold 0 \
  --inner-fold 0
```

### 5.7 Run Multimodal Baseline

현재 multimodal runner는 scaffold 상태다. 구현 시에도 같은 nested split artifact를 읽고, inference input은 image + ordinary tabular metadata만 사용해야 한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_multimodal_experiment \
  --dataset-root data/raw/isic_2024_challenge \
  --outer-fold 0 \
  --inner-fold 0
```

## 6. Required Verification

Split 생성 직후:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py
```

Tabular protocol:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_tabular_baseline_protocol.py
```

전체 관련 smoke verification:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest \
  tests/test_strict_input_export.py \
  tests/test_tabular_baseline_protocol.py \
  tests/test_experiment_operations.py
```

## 7. Audit Points

Split generation outer audit:

```text
export_strict_input_dataset.py records outer balance score and role distribution after outer 5-fold assignment.
```

Split generation inner audit:

```text
For each outer fold, export_strict_input_dataset.py builds inner 4-fold assignments inside cv_train and records inner balance score plus overlap audit.
```

Runner preflight audit:

```text
Tabular/Image/Multimodal runners must read the same nested split artifact and verify patient overlap for the selected (outer_fold, inner_fold).
```

Result summary audit:

```text
Each result summary must record split_protocol, outer_fold, inner_fold, threshold_source, patient_overlap_audit, and triple_balance_audit.
```

## 8. Privileged Signal Rule

`iddx_full_train_only` is candidate-only train-side signal.

Allowed:

```text
training-only auxiliary target
training-only privileged teacher input
training-only representation alignment signal
analysis-only interpretability evidence
```

Disallowed:

```text
ordinary tabular input
validation/test/inference dataloader requirement
threshold selection input
full-data text vectorizer input
```

Paper-facing default inference input is always:

```text
lesion image + ordinary inference-time tabular metadata
```

## 9. Paper-Valid Checklist

Before treating a result as paper-facing, confirm:

```text
patient_id overlap across train/validation/test is zero
preprocessing is fitted on inner_train only
class weights and samplers are built from inner_train only
threshold is selected on inner_validation only
outer_test is used only for final evaluation
iddx_full is not an inference input
required metrics include pAUC above TPR 0.80, AUC, F1, precision, recall, balanced accuracy
result summary includes fold, seed, split source, config path, metric function, threshold source
```
