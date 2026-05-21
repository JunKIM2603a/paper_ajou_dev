# ISIC2024 Strict Input Export

이 문서는 ISIC2024 `strict_input` feature table, train-only `iddx_full` sidecar, patient-level Triple Stratified Nested CV split artifact를 생성하고 검증하는 방법을 정리한다.

## 목적

논문 재현의 시작점은 다음 노트북이다.

```text
notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260514.ipynb
```

이 노트북은 split CSV를 직접 저장하지 않고, 논문 실험이 따라야 할 데이터 입력 계약과 nested CV 감사표를 고정한다. 실제 재현 가능한 산출물 생성은 CLI가 담당한다.

기본 project input contract는 다음과 같다.

```text
image + ordinary inference-time tabular metadata -> malignant probability
```

`iddx_full`은 ordinary inference-time input이 아니다. 후보 실험에서만 train-only privileged supervision signal로 사용할 수 있으므로 strict model input table과 분리한다.

## 주요 파일

`src/isic2024_multimodal/cli/export_strict_input_dataset.py`

- raw `train-metadata.csv`를 읽는다.
- strict input table, iddx sidecar, nested CV split, summary evidence를 쓴다.
- export 전에 source schema를 검증한다.
- export 후 patient overlap, outer/inner split role, `iddx_full` exclusion을 검증한다.

`src/isic2024_multimodal/data/triple_stratified_split.py`

- patient-level profile을 만든다.
- outer 5-fold `cv_test_fold` / `outer_test` assignment를 만든다.
- 각 outer fold의 `cv_train` 내부에서 inner 4-fold `inner_train` / `inner_validation` assignment를 만든다.
- outer와 inner 모두 `assign_triple_stratified_groups`를 사용한다.
- balance objective는 patient 수, row 수, positive row 수, malignant patient 수, sample-count bin을 함께 본다.

`tests/test_strict_input_export.py`

- synthetic dataframe으로 export contract를 빠르게 검증한다.
- raw ISIC data가 없어도 strict input leakage guard, nested split determinism, patient-disjoint behavior를 확인할 수 있다.

`notebooks/isic_2024/isic2024_strict_input_export_audit_20260514.ipynb`

- CLI 산출물을 읽어 audit table을 보여준다.
- dataset을 생성하지 않는다.
- row count, patient overlap, outer/inner fold 분포, malignant count, iddx exclusion을 사람이 확인하는 보고서 역할을 한다.

## 실행 명령

프로젝트 환경에서 `ISIC2024_EXPECTED_CONDA_ENV=paper`와 `PYTHONPATH=./src`를 유지한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

기본 산출물은 다음 위치에 생성된다.

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_official_train_nested_5x4_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

옵션을 바꿔 다른 seed나 fold 수를 만들 수 있다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --seed 42 \
  --outer-folds 5 \
  --inner-folds 4 \
  --sample-count-bins 5
```

테스트 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py
```

## 산출물 계약

`data/processed/isic2024_strict_model_input.csv`

- 포함: `isic_id`, `patient_id`, `lesion_id`, `target`, strict input column 39개.
- 제외: `iddx_full`, `iddx_1`-`iddx_5`, `mel_mitotic_index`, `mel_thick_mm`, `tbp_lv_dnn_lesion_confidence`, `attribution`, `copyright_license`, `image_type`.
- 역할: ordinary inference-time tabular input table.

`data/processed/isic2024_iddx_full_train_only_sidecar.csv`

- 포함: `isic_id`, `patient_id`, `lesion_id`, `target`, `iddx_full_train_only`.
- 역할: train-only privileged supervision candidate sidecar.
- validation, test, inference dataloader가 이 파일을 요구하면 안 된다.

`data/splits/isic2024_official_train_nested_5x4_seed42.csv`

- 최소 포함: `isic_id`, `patient_id`, `lesion_id`, `outer_fold`, `cv_test_fold`, `inner_fold`, `split_role`.
- `cv_test_fold` is the same value as `outer_fold`.
- `split_role` is one of `outer_test`, `inner_train`, `inner_validation`.
- For a selected `(outer_fold, inner_fold)` pair, every `isic_id` has exactly one role row.
- `outer_test`는 최종 평가 전용이다.
- `inner_validation`은 model choice, hyperparameter selection, early stopping, threshold selection, calibration에 사용한다.
- `inner_train`은 preprocessing, class weight, sampler, model parameter를 fit하는 유일한 partition이다.

## 누수 방지 장치

중요한 제어 조건이 실패하면 CLI도 실패한다.

- `cv_train` and `outer_test` patients must be disjoint for every outer fold.
- `inner_train`, `inner_validation`, and `outer_test` patients must be mutually disjoint for every selected nested split.
- Outer and inner assignments must both be produced by the Triple Stratified splitter.
- `iddx_full` and diagnosis/reference columns must not appear in the strict model input table.
- No imputation, scaling, encoding, feature selection, threshold selection, or calibration is fit during export.

Train-only preprocessing belongs in later model training code and must be fit inside `inner_train` only. `outer_test` is never used for model choice or threshold selection.

The export summary may record missingness evidence for strict input features, identifiers, and excluded privileged/reference columns. This evidence is descriptive only. The export CLI must not impute missing values, because imputation parameters must be learned separately inside each training fold.

## Seed 42 현재 근거

생성된 summary 파일은 export 근거를 기록한다.

```text
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

현재 seed 42 export에서 필요한 확인 항목은 다음과 같다.

```text
rows: 401059
patients: 1042
positive_rows: 393
strict_input_columns: 39
outer_folds: 5
inner_folds: 4
patient_disjoint_outer_cv: true
patient_disjoint_inner_cv: true
triple_stratified_outer_folds: true
triple_stratified_inner_folds: true
iddx_full_excluded_from_strict_input: true
diagnosis_reference_columns_excluded_from_strict_input: true
```

`data/processed/**`와 `data/splits/**`는 Git에서 제외되므로 생성된 CSV는 local artifact다. Source code, audit notebook, test, 작은 summary evidence가 Git으로 추적되는 재현 자료다.
