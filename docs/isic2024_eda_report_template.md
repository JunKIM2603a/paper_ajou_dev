# ISIC2024 Tabular EDA Report

## 1. 목적 및 분석 범위

이 문서는 `ISIC2024` tabular 데이터를 대상으로 데이터 구조, 클래스 불균형, 결측 패턴, 주요 범주형/수치형 변수의 분포, leakage 가능성, 그리고 baseline feature set 설계 근거를 정리한다.  
또한 목표 2에서 실행한 tabular baseline 결과와 EDA 해석을 연결하여, 어떤 feature set을 메인 비교 기준으로 삼아야 하는지 논의한다.

### 읽기 가이드

{{reading_guide}}

### 분석 원칙

{{analysis_principles}}

## 2. 데이터 개요

표 1. 데이터셋 개요

{{dataset_overview_table}}

### 해석

{{dataset_overview_interpretation}}

## 3. 클래스 불균형 분석

그림 1. 클래스 분포

![그림 1. 클래스 분포](figures/class_balance.png)

### 해석

{{class_balance_interpretation}}

## 4. 결측 패턴 분석

그림 2. 상위 결측률 컬럼

![그림 2. 상위 결측률 컬럼](figures/missingness_top10.png)

표 2. 상위 결측률 컬럼 요약

{{missingness_table}}

### 해석

{{missingness_interpretation}}

## 5. 범주형 변수 분석

### 5.1 `iddx_1`별 양성 비율

그림 3. `iddx_1`별 양성 비율

![그림 3. iddx_1별 양성 비율](figures/target_rate_iddx1.png)

표 3. `iddx_1`별 양성 비율

{{iddx1_table}}

### 해석

{{iddx1_interpretation}}

### 5.2 `attribution`별 양성 비율

그림 4. `attribution`별 양성 비율

![그림 4. attribution별 양성 비율](figures/target_rate_attribution.png)

표 4. `attribution`별 양성 비율

{{attribution_table}}

### 해석

{{attribution_interpretation}}

## 6. 수치형 변수 분석

### 6.1 `tbp_lv_dnn_lesion_confidence` 분포

그림 5. `tbp_lv_dnn_lesion_confidence` 히스토그램

![그림 5. tbp confidence 히스토그램](figures/tbp_confidence_hist.png)

표 5. 수치형 변수 요약 통계

{{numeric_table}}

### 해석

{{numeric_interpretation}}

## 7. Leakage 후보 분석

표 6. Leakage 후보 및 제외 컬럼

{{leakage_table}}

### 해석

{{leakage_interpretation}}

## 8. Feature Set 설계

### 8.1 실험 설계 관점 요약

{{experiment_design_summary}}

표 7. 컬럼 배치 규칙

{{column_policy_table}}

### 해석

{{column_policy_interpretation}}

### 8.2 Feature set 구성

표 8. Feature set 구성

{{feature_set_table}}

### 해석

{{feature_set_interpretation}}

## 9. Discussion: EDA와 Baseline 결과의 연결

표 9. Tabular baseline 요약

{{baseline_table}}

### Discussion

{{discussion}}

## 10. 결론

{{conclusion}}
