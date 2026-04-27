# ISIC2024 Follow-up `.py` / `.ipynb` 분리 전략

## 1. 목적

`17장` 후속 검증은 반복 실행과 재현성이 중요한 benchmark이므로, 실제 학습과 저장은 `.py`에 남기고, `ipynb`는 결과 해석과 보고서 생성에 집중한다. 이렇게 나누면 notebook의 hidden state에 덜 의존하고, `2 x RTX 4090` 자원도 기존 runner를 통해 안정적으로 사용할 수 있다.

## 2. 기본 원칙

- `.py`는 "같은 입력이면 같은 산출물을 남기는 실행 계층"으로 둔다.
- `.ipynb`는 "이미 저장된 산출물을 읽고 비교하는 분석 계층"으로 둔다.
- notebook에서 학습 루프, 모델 클래스, split 로직을 다시 구현하지 않는다.
- notebook이 실행을 트리거하더라도 기존 `.py` entrypoint를 호출하는 방식으로만 사용한다.

## 3. 역할 분담

| 계층 | 책임 | 현재 파일 |
|---|---|---|
| execution core (`.py`) | data loading, split, model build, metric 계산, trainer | `src/isic2024_multimodal/data.py`, `tabular_data.py`, `models.py`, `metrics.py`, `trainer.py` |
| execution entrypoint (`.py`) | 실제 benchmark 실행, multi-GPU scheduling, artifact 저장 | `src/isic2024_multimodal/run_experiment.py`, `run_tabular_baselines.py`, `run_all_models.py` |
| execution export (`.py`) | leaderboard / HTML / MLflow export | `src/isic2024_multimodal/mlflow_report.py`, `mlflow_html_report.py`, `runtime_env.py` |
| runbook (`.ipynb`) | `.env` 로드, checkpoint readiness, GPU preflight, 실행 명령 정리 | `notebooks/isic_2024/isic2024_followup_validation_17.ipynb` |
| tabular analysis (`.ipynb`) | tabular leaderboard 비교, feature set별 `pauc_above_tpr80` 요약 | `notebooks/isic_2024/isic2024_followup_tabular_analysis_17.ipynb` |
| image analysis (`.ipynb`) | image readiness board, model별 best `pauc_above_tpr80` 비교 | `notebooks/isic_2024/isic2024_followup_image_analysis_17.ipynb` |

## 4. notebook 작성 규칙

- notebook은 `summary.json`, `history.csv`, leaderboard CSV, MLflow export 결과를 읽는 쪽을 기본으로 한다.
- notebook 안에서 epoch loop를 직접 돌리지 않는다.
- notebook 안에서 checkpoint loading 로직을 새로 만들지 않는다.
- notebook이 결과 표나 그림을 만들면 `experiments/evidence/eda/isic_2024/followup_tables` 같은 산출 경로에 저장한다.
- 최종 수치의 source of truth는 항상 `.py` runner가 남긴 `experiments/outputs/`와 `experiments/logs/mlruns/`이다.

## 5. GPU 사용 전략

- 기본 전략은 "한 모델을 여러 GPU로 쪼개는 것"보다 "여러 모델 또는 trial을 GPU 0/1에 병렬 배치하는 것"이다.
- 실제 GPU scheduling은 `src/isic2024_multimodal/run_all_models.py`가 담당한다.
- notebook은 GPU 상태를 확인하고, 적절한 실행 명령을 조합하는 runbook 역할만 맡는다.

## 6. 추천 흐름

1. `isic2024_followup_validation_17.ipynb`에서 `.env`, checkpoint, GPU 상태를 확인한다.
2. image / tabular benchmark는 기존 `.py` runner로 실행한다.
3. 결과는 `experiments/outputs/`와 `experiments/logs/mlruns/`에 저장한다.
4. `isic2024_followup_tabular_analysis_17.ipynb`, `isic2024_followup_image_analysis_17.ipynb`에서 `pauc_above_tpr80` 기준으로 비교한다.
5. 필요하면 이후 notebook에서 error analysis와 논문용 figure/table을 확장한다.

## 7. 이 전략으로 기대하는 효과

- 실행 코드와 해석 코드가 섞이지 않아 유지보수가 쉬워진다.
- 실험 실패 시 notebook 상태와 무관하게 `.py` runner 단위로 재시도할 수 있다.
- section `17`의 follow-up benchmark를 runbook, execution, analysis 세 층으로 깔끔하게 설명할 수 있다.
