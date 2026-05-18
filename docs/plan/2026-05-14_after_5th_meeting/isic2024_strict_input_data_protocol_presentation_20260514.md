# ISIC2024 Strict Input 데이터 처리 발표 요약

작성일: 2026-05-18

참조 노트북:

- `notebooks/isic_2024/isic2024_strict_input_iddx_full_dataset_20260514.ipynb`
- `notebooks/isic_2024/isic2024_strict_input_export_audit_20260514.ipynb`

## 1. 목적

두 노트북은 baseline 시험을 시작하기 전에 어떤 데이터를 어떤 조건으로 사용할지 고정하기 위한 문서다.

- `isic2024_strict_input_iddx_full_dataset_20260514.ipynb`: 데이터 입력 계약 정의
- `isic2024_strict_input_export_audit_20260514.ipynb`: 실제 export 산출물이 계약을 지키는지 감사

핵심 목표는 성능 시험 전에 다음을 먼저 확정하는 것이다.

```text
image + ordinary inference-time tabular metadata -> malignant probability
```

즉, 현재 baseline 단계의 초점은 복잡한 candidate 방법이 아니라, 누수 없는 strict input과 patient-level split을 먼저 안정화하는 것이다.

## 2. 데이터 입력 계약

기본 데이터 pool은 ISIC2024 `train-metadata.csv` 전체다.

| 항목 | 값 |
|---|---:|
| 전체 row | 401,059 |
| 고유 patient | 1,042 |
| malignant row | 393 |
| malignant 비율 | 약 0.098% |
| strict input feature | 39개 |

`strict_input`은 추론 시점에도 사용할 수 있는 ordinary tabular metadata만 포함한다.

- 포함: `isic_id`, `patient_id`, `lesion_id`, `target`, strict metadata 39개
- 제외: `iddx_full`, `iddx_1`-`iddx_5`, diagnosis/reference column, pathology-derived column, schema-only constant column
- 결측 처리, scaling, encoding은 export 단계에서 하지 않고 fold별 training code에서만 fit한다.

`iddx_full`은 baseline 입력이 아니다. 필요한 경우에도 train-only privileged supervision candidate sidecar로만 분리해서 관리한다.

## 3. Split Protocol

기본 split은 patient-level Triple Stratified Nested CV다.

여기서 Triple Stratified는 row를 무작위로 나누는 방식이 아니다. 먼저 `patient_id` 단위 profile을 만들고, 각 patient를 세 가지 기준으로 묶은 뒤 fold마다 비슷하게 배치한다.

**Triple 기준 개요**

| 번호 | 기준 | 왜 필요한가 |
|---:|---|---|
| 1 | malignant 포함 여부 | malignant patient가 특정 fold에 몰리면 rare target 평가가 불안정해진다. |
| 2 | positive row bin | malignant row가 1개인 patient와 여러 개인 patient를 구분해야 positive row 수가 fold마다 비슷해진다. |
| 3 | sample count bin | patient별 row 수 차이가 커서, 큰 patient가 한 fold에 몰리면 row 수와 학습/평가 규모가 흔들린다. |

**기준 1. malignant 포함 여부**

patient 안에 `target=1` row가 하나라도 있으면 `has_malignant=True`, 아니면 `False`로 둔다. 이 기준은 malignant patient 수를 fold마다 최대한 비슷하게 나누기 위한 첫 번째 축이다.

**기준 2. positive row bin**

patient별 malignant row 수를 아래 세 그룹으로 나눈다.

| bin | 의미 |
|---|---|
| `zero` | malignant row가 없는 patient |
| `one` | malignant row가 정확히 1개인 patient |
| `multiple` | malignant row가 2개 이상인 patient |

이 기준은 단순히 malignant patient 수만 맞추는 것보다 더 강하다. malignant patient 수가 같아도 어떤 fold에는 malignant row가 많은 patient가 몰릴 수 있기 때문에, positive row 개수 구간까지 함께 맞춘다.

**기준 3. sample count bin**

patient별 전체 row 수를 5개 구간으로 나눈다. ISIC2024는 patient마다 lesion/sample 수 차이가 크기 때문에, sample이 많은 patient가 특정 fold에 몰리면 patient 수는 비슷해도 row 수가 크게 달라질 수 있다.

**배치 방식**

1. 각 patient에 대해 `has_malignant`, `positive_row_bin`, `sample_count_bin`을 계산한다.
2. 세 값을 합쳐 `triple_stratum`을 만든다.

```text
triple_stratum = has_malignant | positive_row_bin | sample_count_bin
example: 1|pos=one|size=4
```

3. 같은 `triple_stratum`에 속한 patient들을 seed 42로 섞는다.
4. 각 stratum 안에서 outer 5-fold라면 약 20%씩, inner 4-fold라면 약 25%씩 quota에 맞춰 patient를 나눈다.
5. 초기 배치 뒤에는 balance score를 낮추는 patient swap을 수행한다.
6. balance score는 fold별 patient 수, 전체 row 수, positive row 수, malignant patient 수, sample-count bin 분포를 함께 본다.

결과적으로 각 fold는 단순히 patient overlap이 없는 것에서 끝나지 않고, rare malignant 분포와 patient별 sample 수 규모도 함께 비슷하게 유지한다.

노트북과 seed 42 export artifact 기준 결과는 다음과 같다.

| patient-level profile 항목 | 결과 |
|---|---:|
| 전체 patient | 1,042 |
| malignant 포함 patient | 259 |
| `positive_row_bin=zero` | 783 |
| `positive_row_bin=one` | 193 |
| `positive_row_bin=multiple` | 66 |
| `sample_count_bin` 분포 | 209 / 208 / 208 / 208 / 209 |
| unique triple stratum | 15 |

Outer 5-fold의 `outer_test` 분포는 아래처럼 거의 같은 row 수와 positive 비율을 유지한다.

| outer fold | patients | rows | positive rows | positive rate (%) | malignant patients | sample bins | triple strata |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 214 | 80,212 | 78 | 0.09724 | 52 | 5 | 14 |
| 1 | 212 | 80,212 | 78 | 0.09724 | 52 | 5 | 14 |
| 2 | 210 | 80,212 | 79 | 0.09849 | 51 | 5 | 14 |
| 3 | 205 | 80,212 | 79 | 0.09849 | 52 | 5 | 14 |
| 4 | 201 | 80,211 | 79 | 0.09849 | 52 | 5 | 12 |

| balance score 항목 | 결과 |
|---|---:|
| outer 5-fold balance score | 2.5486 |
| inner 4-fold balance score 최소 | 1.2402 |
| inner 4-fold balance score 최대 | 2.2811 |

```text
official_train_pool
(train-metadata.csv 전체: 401,059 rows / 1,042 patients)
        |
        v
+--------------------------------------------------+
| Outer CV: patient-level Triple Stratified 5-fold |
+--------------------------------------------------+
        |
        +-- fold k
        |     |
        |     +-- cv_test_fold == outer_test
        |     |   - fold별 최종 평가 전용
        |     |   - model choice / threshold / calibration 금지
        |     |
        |     +-- 나머지 4 folds == cv_train
        |         |
        |         v
        |   +--------------------------------------------------+
        |   | Inner CV: patient-level Triple Stratified 4-fold |
        |   +--------------------------------------------------+
        |         |
        |         +-- inner_fold j == inner_validation
        |         |   - model choice
        |         |   - hyperparameter / early stopping
        |         |   - threshold / calibration
        |         |
        |         +-- 나머지 inner folds == inner_train
        |             - preprocessing fit
        |             - class weight / sampler fit
        |             - model training
        |
        +-- 모든 outer fold에 대해 같은 절차 반복
```

현재 baseline runner의 실행 단위는 `(outer_fold=k, inner_fold=j)` 하나다. 이 실행은 `inner_train`으로 학습하고, `inner_validation`으로 선택하고, 선택된 설정을 다시 `inner_train`으로 학습한 뒤 `outer_test`를 평가한다.

```text
현재 구현:
outer k, inner j 하나 선택
    inner_train 학습
    inner_validation 선택
    inner_train으로 best 재학습
    outer_test 평가

--all-folds:
    위 과정을 outer x inner = 20번 반복
    요약기는 validation 기준으로 outer별 하나를 고름
    full cv_train refit은 하지 않음
```

따라서 현재 baseline 단계 결과는 `validation-selected nested summary`로 부르는 것이 정확하다. 20개 실행 결과를 그대로 평균하지 않고, outer fold별 validation-selected 대표 실행 5개를 요약한다.

```text
paper-final 후속 절차:
outer_fold k
    outer_test는 최종 평가용으로 계속 잠금
    inner CV 결과로 best hyperparameter 확정
    full cv_train에서 train-only preprocessing + model refit
    outer_test에서 한 번 평가
```

이 paper-final refit은 최종 모델 선정 이후에 별도 수행해야 하는 이상적인 흐름이다.

역할은 다음처럼 고정한다.

| split role | 용도 |
|---|---|
| `inner_train` | preprocessing, class weight, sampler, model fit |
| `inner_validation` | model choice, hyperparameter, early stopping, threshold, calibration |
| `outer_test` | fold별 최종 평가 전용 |

`inner_validation`은 최종 평가용 데이터가 아니라, `outer_test`를 보기 전에 필요한 선택을 끝내기 위한 내부 검증 partition이다. 모델 학습과 전처리 fit은 `inner_train`에서만 수행하고, `inner_validation`은 학습된 모델의 validation probability와 metric을 이용해 선택 결정을 내리는 데 사용한다.

| 결정 항목 | `inner_validation`에서 하는 일 | 현재 baseline 해석 |
|---|---|---|
| 모델 후보 선택 | validation metric으로 trial/model 후보 비교 | `outer_test`를 보기 전에 후보를 고른다. |
| hyperparameter 선택 | learning rate, regularization, model setting 등 trial 비교 | 각 trial은 `inner_train`에서 학습하고 `inner_validation`으로 선택한다. |
| epoch/checkpoint 선택 | image model에서 validation 기준 best epoch/checkpoint 선택 | 실패가 아니라 overfitting을 줄이기 위한 선택이다. |
| threshold 선택 | validation probability에서 F1이 가장 좋은 threshold 선택 | `threshold_source = inner_validation_f1`로 기록한다. |
| calibration | calibration을 구현한다면 여기에서만 fit 가능 | 현재 baseline runner에는 별도 calibration fitting은 없다. |

`inner_validation`에서 하지 않는 것은 다음과 같다.

- imputer, scaler, encoder fit
- feature selection fit
- class weight 또는 sampler 계산
- model parameter 학습
- 최종 논문 성능으로 주장

`outer_test`는 최종 평가에만 사용한다. model choice, threshold selection, calibration, early stopping에는 사용하지 않는다.

## 4. 누수 방지 원칙

논문용 baseline 결과는 다음 조건을 만족해야 한다.

- `patient_id -> lesion_id -> isic_id` grouping을 보존한다.
- outer split에서 `cv_train`과 `outer_test` 사이 patient overlap은 0이어야 한다.
- inner split에서 `inner_train`, `inner_validation`, `outer_test` 사이 patient overlap은 0이어야 한다.
- imputation, scaling, encoding, feature selection, class weight, sampler는 `inner_train`에서만 fit한다.
- threshold-dependent metric의 threshold는 `inner_validation`에서만 선택한다.
- `iddx_full`은 validation, outer test, inference dataloader가 요구하면 안 된다.

**결측 처리 정책**

결측 처리는 데이터 export 시점에 고정하지 않는다. 각 fold의 `inner_train`에서만 imputation, scaling, encoding을 fit하고, `inner_validation`과 `outer_test`에는 그 train-fitted transform만 적용한다.

| 단계 | 정책 |
|---|---|
| export | 결측치를 채우지 않고 raw strict input 값을 보존한다. 결측률과 column profile은 evidence로만 기록한다. |
| training fit | selected `(outer_fold, inner_fold)`의 `inner_train`에서만 imputer, scaler, encoder를 fit한다. |
| validation/test transform | `inner_validation`과 `outer_test`에는 `inner_train`에서 fit한 transform만 적용한다. |

Feature type별 처리는 다음처럼 고정한다.

| feature type | 처리 |
|---|---|
| numeric | `inner_train` median imputation |
| numeric scaling | `inner_train`에서 fit한 `StandardScaler` |
| numeric missing indicator | 현재 `age_approx__missing` 추가 |
| categorical | missing을 `"__missing__"` category로 채움 |
| categorical encoding | 일반 모델은 one-hot encoder를 `inner_train`에서 fit |
| CatBoost categorical | one-hot하지 않고 native categorical feature로 전달 |

감사 노트북 기준으로 현재 seed 42 export는 주요 조건을 통과했다.

| 감사 항목 | 결과 |
|---|---|
| strict input row alignment | pass |
| `iddx_full` strict input 제외 | pass |
| diagnosis/reference column 제외 | pass |
| outer patient overlap | 0 |
| inner patient overlap | 0 |
| outer folds | 5 |
| inner folds | 4 |

## 5. Baseline으로 넘기는 산출물

baseline runner가 읽어야 할 산출물은 아래와 같다.

| 산출물 | 역할 |
|---|---|
| `data/processed/isic2024_strict_model_input.csv` | ordinary metadata baseline input |
| `data/processed/isic2024_iddx_full_train_only_sidecar.csv` | candidate-only train-side signal |
| `data/splits/isic2024_official_train_nested_5x4_seed42.csv` | shared nested CV split artifact |
| `experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json` | split/input audit evidence |

따라서 이후 baseline 시험은 모두 같은 split artifact와 같은 strict input 계약을 사용해야 한다.

Baseline 결과 중 큰 원본 산출물은 Git에 올리지 않고 `experiments/outputs/`에 둔다. Git에는 아래처럼 작게 정리된 결과표를 올리는 흐름이 좋다.

| Git-friendly 산출물 | 역할 |
|---|---|
| `experiments/tables/<family>/<run_group_id>/nested_cv/nested_cv_all_candidates.csv` | 모든 inner 후보 요약 |
| `experiments/tables/<family>/<run_group_id>/nested_cv/nested_cv_outer_selection.csv` | outer fold별 validation-selected 결과 |
| `experiments/tables/<family>/<run_group_id>/nested_cv/nested_cv_metric_summary.csv` | 5개 outer fold의 mean/std/min/max |
| `experiments/tables/<family>/<run_group_id>/nested_cv/nested_cv_summary.md` | 발표/공유용 요약 |

보고 시에는 최소한 다음 정보를 함께 남긴다.

- `split_protocol`, `outer_fold`, `inner_fold`
- threshold source: `inner_validation`
- inference input: image and/or `strict_input`
- `inference_requires_iddx_full=False`
- 필수 metric: pAUC@TPR>=0.80, AUC, F1, precision, recall, balanced accuracy, Average Precision / PR-AUC
