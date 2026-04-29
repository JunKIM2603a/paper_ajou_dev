# ISIC2024 Strict Input Export

이 문서는 ISIC2024 `strict_input` feature table, train-only `iddx_full` sidecar, patient-level split artifact를 생성하고 검증하는 방법을 정리한다.

## Purpose

`notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260427.ipynb`는 dataset contract를 설명하는 문서다. 실제 재현 가능한 산출물 생성은 CLI가 담당한다.

기본 project input contract는 다음과 같다.

```text
image + ordinary inference-time tabular metadata -> malignant probability
```

`iddx_full`은 ordinary inference-time input이 아니다. 후보 실험에서만 train-only privileged supervision signal로 사용할 수 있으므로 strict model input table과 분리한다.

## Main Files

`src/isic2024_multimodal/cli/export_strict_input_dataset.py`

- raw `train-metadata.csv`를 읽는다.
- strict input table, iddx sidecar, holdout split, CV split, summary evidence를 쓴다.
- export 전에 source schema를 검증한다.
- export 후 patient overlap과 `iddx_full` exclusion을 검증한다.

`src/isic2024_multimodal/data/triple_stratified_split.py`

- patient-level profile을 만든다.
- `train_validation_data` / `test_data` holdout assignment를 만든다.
- `train_validation_data` 내부 5-fold `cv_validation_fold` assignment를 만든다.
- balance objective는 patient 수, row 수, positive row 수, malignant patient 수, sample-count bin을 함께 본다.

`tests/test_strict_input_export.py`

- synthetic dataframe으로 export contract를 빠르게 검증한다.
- raw ISIC data가 없어도 strict input leakage guard와 patient-disjoint split behavior를 확인할 수 있다.

`notebooks/isic_2024/isic2024_strict_input_export_audit_20260429.ipynb`

- CLI 산출물을 읽어 audit table을 보여준다.
- dataset을 생성하지 않는다.
- row count, patient overlap, fold 분포, malignant count, iddx exclusion을 사람이 확인하는 보고서 역할을 한다.

## Commands

프로젝트 환경에서 `PYTHONPATH=./src`를 유지한다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

기본 산출물은 다음 위치에 생성된다.

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_train_validation_test_split_seed42.csv
data/splits/isic2024_train_validation_5fold_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

옵션을 바꿔 다른 seed나 fold 수를 만들 수 있다.

```bash
PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --seed 42 \
  --test-size 0.20 \
  --cv-folds 5 \
  --sample-count-bins 5
```

테스트 실행:

```bash
PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py
```

## Output Contracts

`data/processed/isic2024_strict_model_input.csv`

- Contains: `isic_id`, `patient_id`, `lesion_id`, `target`, 39 strict input columns.
- Excludes: `iddx_full`, `iddx_1`-`iddx_5`, `mel_mitotic_index`, `mel_thick_mm`, `tbp_lv_dnn_lesion_confidence`, `attribution`, `copyright_license`, `image_type`.
- Role: ordinary inference-time tabular input table.

`data/processed/isic2024_iddx_full_train_only_sidecar.csv`

- Contains: `isic_id`, `patient_id`, `lesion_id`, `target`, `iddx_full_train_only`.
- Role: train-only privileged supervision candidate sidecar.
- This file must not be required by validation, test, or inference dataloaders.

`data/splits/isic2024_train_validation_test_split_seed42.csv`

- Contains: `isic_id`, `patient_id`, `lesion_id`, `split`.
- `split` is one of `train_validation_data` or `test_data`.
- `test_data` is a locked internal holdout and must not be used for model choice, threshold selection, feature selection, early stopping, or calibration.

`data/splits/isic2024_train_validation_5fold_seed42.csv`

- Contains: `isic_id`, `patient_id`, `lesion_id`, `cv_validation_fold`.
- Contains only rows assigned to `train_validation_data`.
- Each fold defines the validation patients; the remaining train-validation patients are `cv_train`.

## Leakage Controls

The CLI fails if critical controls fail.

- `patient_id` must not overlap between `train_validation_data` and `test_data`.
- For every CV fold, `cv_train` and `cv_validation` patients must be disjoint.
- CV validation patients must not overlap with `test_data` patients.
- `iddx_full` and diagnosis/reference columns must not appear in the strict model input table.
- No imputation, scaling, encoding, feature selection, threshold selection, or calibration is fit during export.

Train-only preprocessing belongs in later model training code and must be fit inside `cv_train` or `final_train` only.

## Current Seed 42 Evidence

The generated summary file records the export evidence:

```text
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

For the current seed-42 export, the key checks are:

```text
rows: 401059
patients: 1042
positive_rows: 393
strict_input_columns: 39
train_validation_test_patient_overlap: 0
patient_disjoint_holdout: true
patient_disjoint_cv: true
iddx_full_excluded_from_strict_input: true
diagnosis_reference_columns_excluded_from_strict_input: true
```

Because `data/processed/**` and `data/splits/**` are ignored by Git, the generated CSVs are local artifacts. The source code, audit notebook, tests, and small summary evidence are the tracked reproducibility material.
