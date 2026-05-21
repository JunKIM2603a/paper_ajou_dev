# ISIC2024 현재 baseline 운영 계획

작성일: 2026-05-21

## 1. 목적

```text
strict ordinary metadata와 lesion image baseline을 같은 patient-level nested CV protocol 아래에서 안정화한다.
```

현재 paper-facing baseline의 목적은 다음 질문에 답하는 것이다.

```text
1. strict input export, split artifact, metric 계산이 코드와 문서에서 일치하는가?
2. strict_main_input tabular metadata만으로 어느 정도 성능이 나오는가?
3. lesion image만으로 어느 정도 성능이 나오는가?
4. image + tabular multimodal baseline을 구현하기 전에 필요한 shared protocol evidence가 준비되었는가?
```

## 2. 현재 baseline 범위

현재 구현 기준의 baseline 범위는 아래와 같다.

| 구분 | 현재 상태 | paper-facing 해석 |
|---|---|---|
| strict export | `export_strict_input_dataset.py` | ordinary inference-time input 계약 생성 |
| split | patient-level Triple Stratified Nested CV, 5 outer x 4 inner | 기본 paper protocol |
| 현재 tabular suite | `xgboost`, `catboost`, `lightgbm`, `ft_transformer`, `ft_transformer_external` | 현재 tabular baseline suite |
| tabular 단독 지원 모델 | `logistic_regression`, `svm`, `mlp` | 단일 runner에서 지원하지만 현재 suite에는 포함하지 않음 |
| 현재 image suite | `resnet50`, `efficientnetv2_s`, `convnextv2_tiny`, `eva02_s`, `vit_b`, `edgenext_s` | 현재 image baseline suite |
| multimodal | `run_multimodal_experiment.py` scaffold | 아직 훈련 미구현, paper-facing 실행 대상 아님 |

현재 baseline에서 말하는 입력과 protocol은 다음과 같다.

```text
입력:
  tabular: strict_main_input ordinary metadata only
  image: lesion image only
  multimodal future: lesion image + strict_main_input ordinary metadata

split:
  patient-level Triple Stratified Nested CV
  outer_fold: final reporting fold
  inner_fold: model selection and threshold selection fold
  outer_test: final evaluation only

threshold:
  inner validation probabilities에서만 선택
  threshold_source = inner_validation_f1

metrics:
  pAUC above TPR 0.80
  AUC
  Average Precision / PR-AUC
  F1
  precision
  recall
  balanced accuracy
  false positive count
  false negative count
```

## 3. 현재 suite에서 제외되는 것

아래 항목은 현재 코드 기준의 기본 baseline suite가 아니다.

| 항목 | 현재 처리 |
|---|---|
| dummy classifier/image sanity baseline | 현재 baseline suite에 포함하지 않음 |
| legacy sklearn tree/boost 후보 | 현재 tabular suite는 GBDT 3종과 FT-Transformer 계열로 고정 |
| 과거 image backbone 후보 목록 | 현재 image suite의 6개 모델로 대체 |
| 직접 concat fusion 계획 | 현재 runner가 미구현이므로 후속 구현 과제로 분리 |
| relaxed/oracle feature set | ordinary inference-time baseline이 아니므로 paper-facing 기본 suite에서 제외 |
| LUPI / `iddx_full` auxiliary | baselines 안정화 이후 candidate-only 연구로 분리 |

## 4. 데이터 처리 원칙

현재 baseline은 strict input export 계약을 따른다.

참조 문서:

```text
docs/eda/isic2024_strict_input_export.md
```

기본 데이터 흐름은 다음과 같다.

```text
raw train metadata/image
  -> strict input export
  -> iddx_full train-only sidecar export
  -> patient-level Triple Stratified Nested CV split artifact
  -> tabular current suite
  -> nested CV summary
  -> image current suite
  -> multimodal implementation after unimodal baselines are stable
```

원칙:

1. Raw data는 `data/raw/isic_2024_challenge/`에서 읽기만 한다.
2. Strict input table은 ordinary inference-time metadata만 포함한다.
3. `iddx_full`, diagnosis text, pathology-derived text는 ordinary model input에 포함하지 않는다.
4. `iddx_full_train_only` sidecar는 기본 baseline에서 사용하지 않는다.
5. 결측치 처리, scaling, encoding, class weight, sampler는 fold train에서만 fit 또는 산출한다.
6. Validation/test에는 train-fitted transform만 적용한다.
7. Tabular와 image baseline은 같은 nested split artifact를 공유한다.
8. Threshold와 model selection은 inner validation에서만 수행한다.
9. `outer_test`는 final metric reporting 전용이며 tuning에 사용하지 않는다.

## 5. 현재 구현 기준 모델 정의

### 5.1 현재 tabular suite

현재 suite config:

```text
experiments/configs/suites/tabular_baselines.json
```

Dataset spec:

```text
experiments/configs/dataset_specs/strict_main_input_v1.json
```

Feature set:

```text
strict_main_input
```

현재 suite 모델:

```text
xgboost
catboost
lightgbm
ft_transformer
ft_transformer_external
```

단일 tabular runner에서 추가로 지원하지만 현재 suite에는 포함하지 않는 모델:

```text
logistic_regression
svm
mlp
```

주의:

```text
ordinary tabular input에는 iddx_full, diagnosis/pathology text, oracle/target-derived feature가 들어가면 안 된다.
tabular preprocessing은 fold train에서만 fit한다.
```

### 5.2 현재 image suite

현재 suite config:

```text
experiments/configs/suites/image_baselines.json
```

Dataset spec:

```text
experiments/configs/dataset_specs/image_preprocessed_v1.json
```

현재 suite 모델:

```text
resnet50
efficientnetv2_s
convnextv2_tiny
eva02_s
vit_b
edgenext_s
```

기본 config 위치:

```text
experiments/configs/image_baselines/<model_name>/config.json
```

주의:

```text
image-only baseline은 lesion image만 inference input으로 사용한다.
metadata/diagnosis text는 image model input으로 사용하지 않는다.
```

### 5.3 Multimodal scaffold와 roadmap

현재 multimodal runner:

```text
src/isic2024_multimodal/cli/run_multimodal_experiment.py
```

현재 상태:

```text
NotImplementedError("Multimodal training is not implemented yet. Use image and tabular baseline CLIs first.")
```

따라서 현재 multimodal config가 있더라도 paper-facing 실행 대상으로 보지 않는다. Multimodal baseline은 tabular/image baseline protocol과 결과 table이 안정화된 뒤, 같은 nested split artifact와 같은 metric function을 공유하도록 구현한다.

`research.md` 기준 후속 구현 우선순위는 다음과 같다.

| ID | 구조 | 역할 | 현재 처리 |
|---|---|---|---|
| `C-Late` | image ensemble + tabular stack + OOF meta-learner | 가장 강한 실전 기준선 | tabular/image suite 결과 안정화 뒤 우선 설계 |
| `C-Early` | pooled image embedding + tabular encoder concat | 최소 neural fusion baseline | late fusion 설계 이후 구현 |
| `C-Middle` | cross-attention 또는 FiLM/GMU conditioning | 대표 deep fusion 비교군 또는 제안법 비교군 | early fusion 이후 구현 |

후속 multimodal 구현의 inference input은 다음으로 제한한다.

```text
lesion image
strict_main_input ordinary metadata
```

`iddx_full` 또는 diagnosis text는 multimodal inference input으로 사용할 수 없다.

공통 구현 규칙:

1. OOF 없는 stacking으로 meta-learner를 학습하지 않는다.
2. Tabular/image OOF prediction은 같은 patient-level nested split artifact에서 만든다.
3. Calibration은 train/inner-validation protocol 안에서만 fit한다.
4. `outer_test`는 final evaluation 전용으로 유지한다.
5. Patient-context, attribution, WB360 appearance metadata 의존도는 ablation으로 분리해 보고한다.

## 6. 실행 순서

### 6.1 Strict input export

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.export_strict_input_dataset
```

생성되는 주요 산출물:

```text
data/processed/isic2024_strict_model_input.csv
data/processed/isic2024_iddx_full_train_only_sidecar.csv
data/splits/isic2024_official_train_nested_5x4_seed42.csv
experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json
```

### 6.2 프로토콜 테스트

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  pytest tests/test_strict_input_export.py tests/test_tabular_baseline_protocol.py
```

### 6.3 Tabular family preflight

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family tabular_baselines \
    --config experiments/configs/suites/tabular_baselines.json \
    --run-group-id tabular_current_v1 \
    --preflight-only
```

### 6.4 Tabular family run

Smoke run:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family tabular_baselines \
    --config experiments/configs/suites/tabular_baselines.json \
    --run-group-id tabular_current_v1_smoke \
    --smoke
```

Dataset spec fold 전체 suite 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family tabular_baselines \
    --config experiments/configs/suites/tabular_baselines.json \
    --run-group-id tabular_current_v1
```

현재 suite로 모든 nested fold 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_all_tabular_models \
    --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
    --feature-sets strict_main_input \
    --split-protocol nested_cv \
    --nested-split-csv data/splits/isic2024_official_train_nested_5x4_seed42.csv \
    --all-folds \
    --run-group-id tabular_current_v1_all_folds
```

### 6.5 Nested CV summary

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.summarize_nested_cv_results \
    --family tabular_baselines \
    --run-group-id tabular_current_v1_all_folds
```

### 6.6 Image family preflight

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family image_baselines \
    --config experiments/configs/suites/image_baselines.json \
    --run-group-id image_current_v1 \
    --preflight-only
```

### 6.7 Image family run

Smoke run:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family image_baselines \
    --config experiments/configs/suites/image_baselines.json \
    --run-group-id image_current_v1_smoke \
    --smoke
```

Dataset spec fold 전체 suite 실행:

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src \
  python -m isic2024_multimodal.cli.run_experiment_family \
    --family image_baselines \
    --config experiments/configs/suites/image_baselines.json \
    --run-group-id image_current_v1
```

현재 image family runner는 suite config의 dataset spec fold를 실행한다. 전체 nested fold 결과가 필요하면 같은 split artifact를 기준으로 outer/inner fold별 실행을 명시적으로 분리해 기록한다.

### 6.8 Multimodal 후속 작업

현재 multimodal runner는 미구현 scaffold다. 아래 config/CLI는 후속 구현 범위 확인용이며, 현재 paper-facing 실행 대상으로 쓰지 않는다.

```text
experiments/configs/suites/multimodal_baselines.json
src/isic2024_multimodal/cli/run_multimodal_experiment.py
```

후속 구현 순서:

1. `C-Late`: 현재 tabular/image suite 결과를 활용해 OOF late fusion stack을 설계한다.
2. `C-Early`: pooled image embedding과 tabular encoder representation을 concat하는 최소 neural fusion을 구현한다.
3. `C-Middle`: cross-attention 또는 FiLM/GMU conditioning을 논문 제안법 또는 직접 비교군으로 구현한다.

Multimodal ablation 후보:

```text
no patient-context
with/without attribution
no WB360/appearance metadata
early vs middle vs late
calibration on/off
```

## 7. 평가 기준

모든 paper-facing baseline 결과는 같은 metric function과 같은 threshold protocol을 사용한다.

필수 metric:

```text
pAUC above TPR 0.80
AUC
F1
precision
recall
balanced accuracy
```

권장 metric:

```text
Average Precision / PR-AUC
false positive count
false negative count
confusion matrix
fold-wise score
mean +/- std
minimum fold score
```

Multimodal 향후 reporting 후보:

```text
pAUC above TPR 0.88 sensitivity analysis
calibration metrics such as Brier score and ECE
Top-K retrieval sensitivity
decision-curve net benefit
```

Threshold-dependent metric은 validation에서 선택한 threshold만 사용한다.

```text
threshold_source = inner_validation_f1
```

Outer test fold는 final reporting 전용이다.

```text
outer_test는 threshold selection, model choice, calibration, feature selection에 사용하지 않는다.
```

## 8. 산출물 위치

실행 산출물:

```text
experiments/outputs/tabular_baselines/
experiments/outputs/image_baselines/
```

Paper-ready 요약 표:

```text
experiments/tables/tabular_baselines/
experiments/tables/image_baselines/
```

Validation/protocol 근거:

```text
experiments/evidence/validation_protocol/
experiments/evidence/eda/isic_2024/
```

## 9. 현재 baseline 완료 기준

현재 baseline 안정화는 아래가 충족될 때 완료로 본다.

1. Strict input export가 현재 code/test와 일치한다.
2. 모든 결과가 patient-level Triple Stratified Nested CV split을 사용한다.
3. Patient overlap audit이 0이다.
4. `iddx_full`과 diagnosis/reference columns가 ordinary input에 없다.
5. Preprocessing, class weight, sampler는 fold train에서만 fit 또는 산출된다.
6. Threshold는 inner validation에서만 선택된다.
7. `outer_test`는 final metric reporting에만 쓰인다.
8. Fold, seed, config path, split source, threshold source가 결과에 기록된다.
9. Tabular/image 결과가 같은 metric table contract로 정리된다.
10. Multimodal 구현은 위 baseline evidence가 준비된 뒤 시작한다.

## 10. 한 줄 결론

현재 baseline 계획은 다음 한 문장으로 요약한다.

```text
strict_main_input tabular suite and six image backbones under the same patient-level 5x4 nested CV protocol, with multimodal kept as a follow-up scaffold until implemented.
```
