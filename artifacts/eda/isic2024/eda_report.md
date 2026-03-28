# ISIC2024 Tabular EDA Report

## 1. 목적 및 분석 범위

이 문서는 `ISIC2024` tabular 데이터를 대상으로 데이터 구조, 클래스 불균형, 결측 패턴, 주요 범주형/수치형 변수의 분포, leakage 가능성, 그리고 baseline feature set 설계 근거를 정리한다.  
또한 목표 2에서 실행한 tabular baseline 결과와 EDA 해석을 연결하여, 어떤 feature set을 메인 비교 기준으로 삼아야 하는지 논의한다.

## 2. 데이터 개요

표 1. 데이터셋 개요

| 항목 | 값 |
| --- | --- |
| dataset_root | /home/junkim2603a/proj/paper_ajou_dev/dataset/ISIC2024 |
| rows | 401059 |
| target_column | malignant |
| positive_count | 393 |
| negative_count | 400666 |
| positive_ratio | 0.000980 |
| column_count | 16 |

### 해석

본 데이터셋은 총 `401,059`개의 표본으로 구성되어 있으며, 이 중 양성 비율은 `0.000980`에 불과하다. 이는 일반적인 분류 데이터셋과 비교해도 매우 극단적인 불균형 조건에 해당한다. 따라서 이후 baseline 결과를 해석할 때 단순 정확도보다는 양성 탐지 능력을 반영하는 지표를 우선적으로 살펴봐야 한다.

또한 전체 컬럼 수는 많지 않지만, 컬럼의 성격은 균질하지 않다. 일부 변수는 메타데이터 수준의 보조 정보인 반면, 일부 변수는 사실상 진단 결과와 매우 가까운 의미를 가진다. 이 때문에 이번 EDA의 핵심 목적은 단순 분포 요약이 아니라, 어떤 컬럼을 메인 baseline feature로 허용할 수 있는지에 대한 판단 근거를 만드는 데 있다.

## 3. 클래스 불균형 분석

그림 1. 클래스 분포

![그림 1. 클래스 분포](figures/class_balance.png)

### 해석

음성 표본은 `400,666`건인 반면 양성 표본은 `393`건에 불과하다. 이 차이는 모델이 단순히 음성만 예측해도 매우 높은 정확도를 얻을 수 있음을 의미한다. 즉, 이 문제에서 정확도는 모델이 실제로 병변을 잘 탐지하는지를 보여주는 대표 지표가 될 수 없다.

따라서 목표 2에서 구성한 tabular baseline은 `best_average_precision`, `balanced_accuracy`, `recall`을 함께 확인하도록 설계했다. 이는 모델이 얼마나 많은 양성을 실제로 포착하는지, 그리고 예측 점수의 순위가 얼마나 유의미한지를 동시에 보기 위함이다.

## 4. 결측 패턴 분석

그림 2. 상위 결측률 컬럼

![그림 2. 상위 결측률 컬럼](figures/missingness_top10.png)

표 2. 상위 결측률 컬럼 요약

| column | missing_count | missing_ratio |
| --- | --- | --- |
| iddx_5 | 401058 | 0.999998 |
| mel_mitotic_index | 401006 | 0.999868 |
| mel_thick_mm | 400996 | 0.999843 |
| iddx_4 | 400508 | 0.998626 |
| iddx_3 | 399994 | 0.997345 |
| iddx_2 | 399991 | 0.997337 |
| lesion_id | 379001 | 0.945001 |
| attribution | 0 | 0.0 |
| copyright_license | 0 | 0.0 |
| iddx_1 | 0 | 0.0 |

### 해석

`iddx_5`의 결측률은 `0.999998`이다. `mel_mitotic_index`의 결측률은 `0.999868`이다. `mel_thick_mm`의 결측률은 `0.999843`이다. `iddx_4`의 결측률은 `0.998626`이다. `iddx_3`의 결측률은 `0.997345`이다. 상위 결측 컬럼 대부분은 `iddx_*` 후반부와 `mel_*` 계열로 나타났다. 이 패턴은 두 가지 해석을 가능하게 한다. 첫째, 이러한 변수는 데이터셋 전반에서 관측 가능한 일반 변수라기보다 특정 상황에서만 기록되는 후속 진단 정보일 가능성이 높다. 둘째, 실제 baseline feature로 사용할 경우 결측 처리 자체가 결과를 왜곡할 수 있다.

`lesion_id` 역시 결측이 매우 많지만, 완전히 무시하기에는 아까운 변수다. 실제 baseline 실험에서 `relaxed` 세트가 `strict`보다 크게 좋아지는 양상이 확인되므로, 이 변수는 단순 메타데이터 이상의 정보를 담고 있을 가능성이 있다. 다만 바로 메인 baseline에 포함하기보다는, `strict`와 분리된 보조 실험 세트로 관리하는 것이 해석상 안전하다.

## 5. 범주형 변수 분석

### 5.1 `iddx_1`별 양성 비율

그림 3. `iddx_1`별 양성 비율

![그림 3. iddx_1별 양성 비율](figures/target_rate_iddx1.png)

표 3. `iddx_1`별 양성 비율

| iddx_1 | count | positive_count | positive_ratio |
| --- | --- | --- | --- |
| Benign | 400552 | 0 | 0.0 |
| Malignant | 393 | 393 | 1.0 |
| Indeterminate | 114 | 0 | 0.0 |

### 해석

`iddx_1=Malignant`의 양성 비율은 `1.000000`이고, `iddx_1=Benign`은 `0.000000`이다. 이 값은 단순 상관 수준을 넘어, `iddx_1`이 사실상 타깃에 대한 직접적인 정보를 포함하고 있음을 보여준다.

즉, 이 변수는 메타데이터라기보다 이미 정리된 진단 판단 결과에 가깝다. 따라서 `iddx_1`을 일반 baseline feature에 포함하면 모델이 입력 데이터를 학습하는 것이 아니라, 이미 주어진 정답 힌트를 활용하는 구조가 된다. 이 때문에 본 프로젝트에서는 `iddx_1`을 `oracle` 세트에만 포함시키고, 메인 비교에서는 제외하는 것이 타당하다.

### 5.2 `attribution`별 양성 비율

그림 4. `attribution`별 양성 비율

![그림 4. attribution별 양성 비율](figures/target_rate_attribution.png)

표 4. `attribution`별 양성 비율

| attribution | count | positive_count | positive_ratio |
| --- | --- | --- | --- |
| Frazer Institute, The University of Queensland, Dermatology Research Centre | 51768 | 81 | 0.001565 |
| Memorial Sloan Kettering Cancer Center | 129068 | 174 | 0.001348 |
| ACEMID MIA | 28665 | 33 | 0.001151 |
| ViDIR Group, Department of Dermatology, Medical University of Vienna | 12640 | 14 | 0.001108 |
| Department of Dermatology, University of Athens, Andreas Syggros Hospital of Skin and Venereal Diseases, Alexander Stratigos, Konstantinos Liopyris | 7976 | 6 | 0.000752 |
| Department of Dermatology, Hospital Clínic de Barcelona | 105724 | 72 | 0.000681 |
| University Hospital of Basel | 65218 | 13 | 0.000199 |

### 해석

기관별 양성 비율은 최소 `0.000199`에서 최대 `0.001565`까지 차이가 난다. 이는 수집 기관에 따라 표본 구성과 난이도가 다를 수 있음을 시사한다. 다시 말해 `attribution`은 병변의 본질적 성질을 설명하는 변수라기보다, 데이터가 어떤 환경에서 수집되었는지를 반영하는 변수일 가능성이 높다.

`attribution`은 `iddx_*`처럼 직접적인 leakage 컬럼으로 보기는 어렵지만, 분포 차이를 통해 모델 성능에 간접적인 영향을 줄 수 있다. 따라서 완전 제외보다는 `strict` 세트에 포함하되, 결과 해석 시 기관별 편향 가능성을 항상 함께 고려하는 것이 바람직하다.

## 6. 수치형 변수 분석

### 6.1 `tbp_lv_dnn_lesion_confidence` 분포

그림 5. `tbp_lv_dnn_lesion_confidence` 히스토그램

![그림 5. tbp confidence 히스토그램](figures/tbp_confidence_hist.png)

표 5. 수치형 변수 요약 통계

| column | group | count | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| mel_mitotic_index | all | 0 |  |  |  |  |
| mel_mitotic_index | target_0 | 0 |  |  |  |  |
| mel_mitotic_index | target_1 | 0 |  |  |  |  |
| mel_thick_mm | all | 63 | 0.670952 | 0.4 | 0.2 | 5.0 |
| mel_thick_mm | target_0 | 0 |  |  |  |  |
| mel_thick_mm | target_1 | 63 | 0.670952 | 0.4 | 0.2 | 5.0 |
| tbp_lv_dnn_lesion_confidence | all | 401059 | 97.162204 | 99.994588 | 0.0 | 100.0 |
| tbp_lv_dnn_lesion_confidence | target_0 | 400666 | 97.177634 | 99.99461 | 0.0 | 100.0 |
| tbp_lv_dnn_lesion_confidence | target_1 | 393 | 81.431493 | 99.68489 | 2e-06 | 100.0 |

### 해석

`tbp_lv_dnn_lesion_confidence`의 평균은 음성 `97.177634`, 양성 `81.431493`로 차이가 나타난다. 평균 차이만으로 모든 것이 설명되지는 않지만, 최소한 이 컬럼이 양성과 음성을 구분하는 데 일정 수준의 신호를 제공하고 있음을 보여준다. 현재 `strict` 세트에서 가장 핵심적인 수치형 변수로 남는 이유도 여기에 있다.

반면 `mel_thick_mm`는 양성에서만 유효값 `63`개가 관측되었다. 이런 패턴은 모델이 병변 특성을 학습한다기보다, 후속 진단 과정에서만 기록된 정보를 통해 양성을 맞히게 만들 수 있다. 따라서 이 변수는 설명용으로는 의미가 있지만, 메인 baseline feature로 사용하기에는 위험하다. `mel_mitotic_index` 역시 유효값이 거의 없어 실제 학습 feature로서는 적절하지 않다.

## 7. Leakage 후보 분석

표 6. Leakage 후보 및 제외 컬럼

| 구분 | 컬럼 |
| --- | --- |
| excluded_columns | image_exists, image_path, isic_id |
| high_leakage_risk_columns | iddx_1, iddx_2, iddx_3, iddx_4, iddx_5, iddx_full, mel_mitotic_index, mel_thick_mm |

### 해석

`iddx_*`, `iddx_full`은 계층형 진단 정보를 직접 담고 있어 leakage 위험이 높다. 이 변수들을 포함했을 때 모델 성능이 과도하게 좋아진다면, 그것은 모델이 실제 예측 능력을 가진 것이 아니라 정답에 가까운 정보를 받아들였기 때문일 수 있다.

`mel_thick_mm`, `mel_mitotic_index` 역시 관측 패턴이 일반형 입력 변수와 다르다. 특히 유효값이 양성에만 집중되거나 거의 존재하지 않는 경우, 메인 baseline에서 사용하면 설명 가능성과 일반화 가능성을 동시에 해친다. 따라서 본 프로젝트는 `strict`를 메인 비교 세트로, `oracle`을 leakage 상한선 확인용 세트로 명확히 분리한다.

## 8. Feature Set 설계

표 7. Feature set 구성

| feature_set | num_columns | columns |
| --- | --- | --- |
| strict | 3 | attribution, copyright_license, tbp_lv_dnn_lesion_confidence |
| relaxed | 4 | attribution, copyright_license, lesion_id, tbp_lv_dnn_lesion_confidence |
| oracle | 12 | attribution, copyright_license, iddx_1, iddx_2, iddx_3, iddx_4, iddx_5, iddx_full, lesion_id, mel_mitotic_index, mel_thick_mm, tbp_lv_dnn_lesion_confidence |

### 해석

`strict`는 현실형 비교 기준으로, 실제 메인 결과표에 가장 적합하다. `relaxed`는 주의가 필요한 보조 정보까지 일부 포함하여, 어떤 컬럼이 성능을 얼마나 끌어올리는지 탐색하는 실험 세트다. `oracle`은 진단 계열 변수까지 포함하는 상한선 세트로, 모델 성능 자체보다는 leakage 영향의 크기를 보여주는 참고 기준으로 이해해야 한다.

## 9. Discussion: EDA와 Baseline 결과의 연결

표 8. Tabular baseline 요약

| model_name | feature_set | best_average_precision | balanced_accuracy | recall | auc_roc |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | oracle | 1.0 | 1.0 | 1.0 | 1.0 |
| svm | oracle | 0.9999999999999998 | 1.0 | 1.0 | 1.0 |
| mlp | strict | 0.0211098448202818 | 0.5 | 0.0 | 0.7327542643898821 |
| logistic_regression | strict | 0.0114549881539307 | 0.6599033063228585 | 0.5949367088607594 | 0.7280103315579621 |
| svm | strict | 0.0111597661741734 | 0.6611824297801108 | 0.5949367088607594 | 0.7256022305954326 |
| xgboost | strict | 0.0087553395735363 | 0.621651630746163 | 0.430379746835443 | 0.6551228835225994 |
| catboost | strict | 0.0077176374618058 | 0.6729712169973117 | 0.6075949367088608 | 0.7294869115538456 |

### Discussion

`strict` 세트에서는 `mlp`가 가장 높은 `best_average_precision=0.021110`를 기록했다. 이는 극단적 불균형 환경에서도 제한된 메타데이터와 신뢰도 점수만으로 일정 수준의 순위화가 가능함을 보여준다.

`oracle` 세트에서는 `logistic_regression`가 `best_average_precision=1.000000`를 기록했다. 현재 결과에서 `oracle` 성능이 거의 완벽해지는 현상은, EDA에서 지적한 `iddx_*` 계열 leakage 위험이 실제 실험 결과로도 재확인되었음을 의미한다.

종합하면, EDA는 단순히 분포를 설명하는 단계에 그치지 않고, 어떤 feature set이 메인 baseline 비교 기준이 되어야 하는지 직접적인 실험 설계 근거를 제공했다. 본 프로젝트에서는 `strict`를 메인 비교 기준으로 유지하고, `relaxed`와 `oracle`은 보조 해석 및 leakage 확인용으로 제한하는 것이 가장 일관된 선택이다.

## 10. 결론

본 EDA는 `ISIC2024` tabular 데이터가 단순한 메타데이터 분류 문제가 아니라, 극단적 클래스 불균형과 강한 leakage 후보가 동시에 존재하는 민감한 실험 환경임을 보여주었다. 특히 `iddx_1`, `iddx_full`, `mel_thick_mm` 계열은 분포와 baseline 결과 모두에서 비정상적으로 강한 신호를 보였기 때문에, 메인 비교 실험에 그대로 사용하는 것은 적절하지 않다.

따라서 현재 시점에서 가장 타당한 메인 baseline 기준은 `strict` feature set이다. `relaxed`와 `oracle`은 성능 향상 자체를 보고하기보다는, 어떤 컬럼이 결과를 과도하게 좋게 만드는지를 보여주는 보조 분석으로 활용하는 것이 적절하다. 이 결론은 이후 image baseline 결과와 tabular baseline 결과를 공정하게 비교할 때도 중요한 기준점이 된다.
