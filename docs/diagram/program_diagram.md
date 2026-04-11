# 프로그램 동작 다이어그램

이 문서는 현재 저장소의 주요 흐름을 `ISIC2024 tabular`와 `image baseline 구조` 중심으로 설명합니다.

## 1. Tabular EDA 흐름

```mermaid
flowchart TD
    A[ISIC_2024_Training_GroundTruth.csv] --> C[tabular_data.py]
    B[ISIC_2024_Training_Supplement.csv] --> C
    C --> D[isic_id 기준 병합]
    D --> E[isic2024_tabular_eda.py]
    E --> F[dataset_overview.json]
    E --> G[missingness_summary.csv]
    E --> H[target_rate_by_*.csv]
    E --> I[report.md]
```

## 2. Feature Set 추천 흐름

```mermaid
flowchart LR
    A[EDA 산출물] --> B[define_tabular_feature_sets.py]
    B --> C[feature_sets_recommended.json]
    C --> D[strict]
    C --> E[relaxed]
    C --> F[oracle]
```

## 3. Tabular Baseline 실행 흐름

```mermaid
flowchart TD
    A[run_tabular_baselines.py] --> B[pandas 병합 로드]
    B --> C[feature set 선택]
    C --> D[train / val / test stratified split]
    D --> E[전처리 파이프라인]
    E --> F[baseline 모델 학습]
    F --> G[val / test metric 계산]
    G --> H[summary.json 저장]
    G --> I[MLflow child run 기록]
    I --> J[모델별 best trial 선택]
    J --> K[MLflow parent run 기록]
```

## 4. Tabular 모델 분기

```mermaid
flowchart TD
    A[feature set + model_name] --> B{모델 종류}
    B -->|sklearn| C[Logistic Regression / SVM / MLP]
    B -->|xgboost| D[XGBoost]
    B -->|catboost| E[CatBoost]
    C --> F[ColumnTransformer]
    D --> F
    E --> G[범주형 직접 처리]
    F --> H[평가]
    G --> H
```

## 5. Tabular MLflow 구조

```mermaid
flowchart TD
    A[Experiment: ISIC2024-Tabular-Benchmark] --> B[Parent Run role=model_parent]
    B --> C[Trial 001 strict]
    B --> D[Trial 002 relaxed]
    B --> E[Trial 003 oracle]
    C --> F[test_average_precision 등]
    D --> G[test_average_precision 등]
    E --> H[test_average_precision 등]
    B --> I[best_average_precision]
    B --> J[best_feature_set]
    B --> K[best_child_run_name]
```

## 6. 리포트 생성 흐름

```mermaid
flowchart LR
    A[MLflow runs] --> B[mlflow_report.py]
    A --> C[mlflow_html_report.py]
    B --> D[mlflow_leaderboard.csv]
    C --> E[mlflow_report.html]
```

## 7. Image Baseline 구조

```mermaid
flowchart TB
    A[src/image_baselines/모델명/config.json] --> D[run_experiment.py]
    B[src/isic2024_benchmark/run_all_models.py] --> D
    D --> E[data.py]
    E --> E1[manifest build]
    E1 --> E2[group-aware train/val/test split]
    D --> F[models.py]
    D --> G[trainer.py]
    G --> H[artifacts/모델명/trial명]
    D --> I[mlruns]
    D --> J[smoke options]
```

## 8. 현재 해석 포인트

```mermaid
mindmap
  root((현재 해석))
    strict
      현실형 baseline
      메인 비교표 권장
    relaxed
      보조 정보 포함
      성능 상승 여부 확인
    oracle
      진단 정보 포함
      leakage 확인용
    목표 3
      image loader 전환 완료
      smoke 검증 완료
      전체 모델 baseline 실행 남음
```
