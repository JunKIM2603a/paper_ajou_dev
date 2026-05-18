# ISIC2024 Reproducibility Guide

이 문서는 여러 컴퓨터에서 git clone 후 현재까지의 논문 실험 protocol을 재현하기 위한 시작 순서와 검증 기준을 정리한다.

## Starting Point

반드시 아래 노트북부터 확인한다.

```text
notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260514.ipynb
```

이 노트북은 논문 실험의 데이터 입력 계약과 split protocol을 고정하는 시작점이다. 논문 본문 초안이 아니라, 이후 tabular, image, multimodal 실험이 paper-valid가 되기 위해 지켜야 하는 기준 문서다.

핵심 용어는 다음과 같다.

| term | meaning |
|---|---|
| `official_train_pool` | ISIC2024 `train-metadata.csv` 전체 |
| `cv_test_fold` | outer fold id이며 `outer_test`와 같은 의미 |
| `cv_train` | 현재 outer test fold를 제외한 outer train pool |
| `inner_train` | `cv_train` 내부에서 다시 나눈 inner training partition |
| `inner_validation` | model choice, hyperparameter, early stopping, threshold, calibration 전용 partition |
| `outer_test` | 최종 평가 전용 partition |

## Protocol Contract

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

`split_role`은 다음 세 값만 허용한다.

```text
outer_test
inner_train
inner_validation
```

`outer_test`는 model choice, hyperparameter search, early stopping, threshold selection, calibration fitting, preprocessing fitting에 사용할 수 없다. `inner_train`만 fit에 사용하고, `inner_validation`만 선택과 threshold 결정에 사용한다.

## Reproduction Order

1. Repository와 환경 준비

```bash
git clone <repo-url>
cd paper_ajou_dev
conda activate paper
```

새 컴퓨터에서는 ISIC2024 raw data를 로컬에 준비해야 한다. Raw data는 git으로 추적하지 않는다.

```text
data/raw/isic_2024_challenge/
```

현재 일부 로컬 명령은 `data/raw`도 dataset root로 받을 수 있다. 새 환경에서는 `data/raw/isic_2024_challenge/`를 기본 배치로 맞추는 것을 권장한다.

2. 시작 노트북 확인

```text
notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260514.ipynb
```

Section 6에서 `cv_train`, `cv_test_fold`, `inner_train`, `inner_validation`, `outer_test`가 어떻게 정의되는지 확인한다. 특히 outer/inner patient overlap audit이 모두 0이어야 한다.

3. strict input / iddx sidecar / nested split 생성

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw/isic_2024_challenge \
  --seed 42 \
  --outer-folds 5 \
  --inner-folds 4
```

생성되는 주요 artifact:

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_official_train_nested_5x4_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

4. export audit notebook 실행

```text
notebooks/isic_2024/isic2024_strict_input_export_audit_20260514.ipynb
```

이 노트북은 생성된 CSV를 읽어서 strict input column, `iddx_full` exclusion, nested split role, patient overlap, fold distribution을 확인한다.

5. Tabular baseline

Preflight:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --outer-fold 0 \
  --inner-fold 0 \
  --preflight-only
```

Run:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --outer-fold 0 \
  --inner-fold 0
```

Fold-wise paper result는 outer fold를 반복해서 만든다.

```text
outer_fold = 0, 1, 2, 3, 4
inner_fold = 0, 1, 2, 3
```

단일 runner는 지정한 `(outer_fold, inner_fold)` 조합 하나를 실행한다. 전체 nested fold 조합을 자동 실행하려면 다음 orchestrator를 사용한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw/isic_2024_challenge \
  --models logistic_regression \
  --feature-sets strict_main_input \
  --all-folds
```

`--all-folds`는 5x4 nested split 기준으로 20개 `(outer_fold, inner_fold)` 실행을 만든다. 각 실행은 `inner_train`으로 trial을 학습하고 `inner_validation`으로 best hyperparameter와 threshold를 선택한 뒤, 같은 `inner_train`으로 best 설정을 다시 학습하고 `outer_test`를 평가한다.

현재 nested summary orchestration은 `summarize_nested_cv_results`로 수행한다. 이 도구는 20개 실행의 `summary.json`을 읽고 validation metric 기준으로 outer fold별 대표 실행을 골라 `validation-selected nested summary`를 만든다. Full `cv_train` refit orchestration은 아직 없으므로, final paper model 확정 후 별도 단계로 수행해야 한다.

6. Image baseline

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --dataset-root data/raw/isic_2024_challenge \
  --config experiments/configs/image_baselines/resnet50/config.json \
  --outer-fold 0 \
  --inner-fold 0
```

Image runner도 tabular runner와 같은 nested split artifact를 읽어야 한다. Image manifest는 `isic_id`로 split artifact에 join된다.

7. Multimodal baseline

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_multimodal_experiment \
  --dataset-root data/raw/isic_2024_challenge \
  --outer-fold 0 \
  --inner-fold 0
```

현재 multimodal runner는 scaffold 상태다. 구현 시에도 같은 nested split artifact를 읽고, inference input은 image + ordinary tabular metadata만 사용해야 한다.

8. Result table / report

결과 summary에는 최소한 다음 metadata가 있어야 paper-facing으로 볼 수 있다.

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

## Required Verification

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

## Triple Stratified Review Points

1. Split generation outer audit

`export_strict_input_dataset.py`가 outer 5-fold assignment 직후 outer balance score와 role distribution을 summary에 기록한다.

2. Split generation inner audit

각 outer fold의 `cv_train` 안에서 inner 4-fold assignment를 만들고 inner balance score와 overlap audit을 기록한다.

3. Runner preflight audit

Tabular/Image/Multimodal runner는 training 전에 같은 nested split artifact를 읽고 selected `(outer_fold, inner_fold)`의 patient overlap audit을 확인한다.

4. Result summary audit

각 result summary는 `split_protocol`, `outer_fold`, `inner_fold`, `threshold_source`, `patient_overlap_audit`, `triple_balance_audit`를 기록한다.

## Privileged Signal Rule

`iddx_full_train_only`는 candidate-only train-side signal이다.

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

Paper-facing default inference input은 항상 다음으로 유지한다.

```text
lesion image + ordinary inference-time tabular metadata
```
