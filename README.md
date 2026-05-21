# ISIC2024 멀티모달 연구 워크스페이스

이 저장소는 ISIC2024 기반 초희귀 malignant target 분류 연구를 위한 작업 공간이다. 현재 기본 연구 방향은 다음 입력 계약을 지키는 baseline과 multimodal 모델을 안정화하는 것이다.

```text
lesion image + ordinary inference-time tabular metadata -> malignant probability
```

`iddx_full`, diagnosis text, pathology-derived field는 ordinary inference-time input이 아니다. 이런 정보는 LUPI 또는 privileged supervision candidate로 명시한 경우에만 training-only signal로 다룬다.

자세한 문서 길잡이는 [docs/README.md](docs/README.md)를 먼저 본다.

## Critical Protocol Rules

- 모든 paper-facing 실험은 patient-level split을 사용한다.
- 같은 `patient_id`가 train, validation, test partition 사이에 겹치면 안 된다.
- imputation, scaling, encoding, feature selection, class weight, sampler는 training partition에서만 fit한다.
- threshold, model choice, hyperparameter, early stopping, calibration은 validation partition에서만 선택한다.
- `outer_test`는 최종 평가 전용이며 선택 과정에 사용하지 않는다.
- `iddx_full`은 validation, test, inference dataloader가 요구하면 안 된다.

## Repository Map

```text
src/isic2024_multimodal/          # Python package, CLI, model/training/evaluation code
notebooks/isic_2024/              # ISIC2024 EDA and audit notebooks
experiments/configs/              # tracked experiment configs and dataset specs
experiments/evidence/             # small tracked protocol and EDA evidence
experiments/tables/               # small tracked result tables and summaries
experiments/outputs/              # generated outputs, checkpoints, caches, ignored by Git
experiments/logs/                 # MLflow and run logs, ignored by Git
data/raw/isic_2024_challenge/     # local raw ISIC2024 data, ignored by Git
data/processed/                   # generated datasets, ignored by Git
data/splits/                      # generated split artifacts, ignored by Git
docs/                             # protocol, reproducibility, plans, references, reports
```

## Start Here

모든 명령은 `paper` conda 환경과 `PYTHONPATH=./src`를 기준으로 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m <module>
```

### 1. Strict Input과 Nested Split 생성

권장 raw data 위치는 `data/raw/isic_2024_challenge/`다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw/isic_2024_challenge
```

생성되는 CSV는 `data/processed/`, `data/splits/` 아래에 저장되며 Git에 올리지 않는다. 추적 가능한 작은 audit evidence는 `experiments/evidence/validation_protocol/`에 남긴다.

상세 설명: [docs/eda/isic2024_strict_input_export.md](docs/eda/isic2024_strict_input_export.md)

### 2. Protocol Test

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest \
  tests/test_strict_input_export.py \
  tests/test_tabular_baseline_protocol.py
```

전체 운영 재현 순서: [docs/reproducibility.md](docs/reproducibility.md)

### 3. Tabular Baseline Smoke Run

Family runner를 우선 사용한다. 이 경로는 suite config, dataset spec, output/table 경로, registry를 함께 맞춘다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_experiment_family \
  --family tabular_baselines \
  --config experiments/configs/suites/tabular_baselines.json \
  --run-group-id tabular_smoke \
  --smoke \
  --devices 0
```

Tabular baseline operating notes:

- GPU/CPU: 기본은 GPU 우선 `auto`이고, CUDA 초기화가 실패하면 CPU로 fallback한다. All-model runner에서 CPU를 강제하려면 `--device-policy cpu`, 단일 runner에서 CPU를 강제하려면 `--device cpu`를 사용한다.
- Safe reset: tabular family만 초기화할 때는 `run_experiment_family --family tabular_baselines --reset-family-output`을 사용한다. 이 경로는 raw data, split artifact, registry를 삭제하지 않는다.
- output/table evidence: 큰 산출물은 `experiments/outputs/tabular_baselines/<run_group_id>/`, 작은 결과표는 `experiments/tables/tabular_baselines/<run_group_id>/`, selection registry는 `experiments/registry/selections/`에 둔다.
- All-folds/nested summary: `--all-folds`는 5x4 nested split에서 20개 실행을 만들고, 아래 `summarize_nested_cv_results` 명령으로 validation-selected nested summary를 만든다.

세부 모델별 backend와 protocol 설명은 [docs/eda/isic2024_tabular_baselines.md](docs/eda/isic2024_tabular_baselines.md)에 둔다.

### 4. Current Nested CV Summary

현재 baseline runner는 `(outer_fold, inner_fold)` 하나를 실행 단위로 사용한다.

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

20개 실행 결과를 Git-friendly summary로 정리할 때는 다음 도구를 쓴다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.summarize_nested_cv_results \
  --family tabular_baselines \
  --run-group-id <run_group_id>
```

이 결과는 `validation-selected nested summary`다. 논문 final model 확정 후에는 outer fold별 best hyperparameter를 확정하고, full `cv_train`에서 train-only preprocessing과 model을 다시 fit한 뒤, `outer_test`에서 한 번 평가하는 paper-final refit 절차가 별도로 필요하다.

## Where To Read More

| 문서 | 역할 |
|---|---|
| [docs/README.md](docs/README.md) | 문서 전체 길잡이 |
| [docs/reproducibility.md](docs/reproducibility.md) | 새 환경에서 protocol을 재현하는 순서 |
| [docs/eda/isic2024_strict_input_export.md](docs/eda/isic2024_strict_input_export.md) | strict input, `iddx_full` sidecar, nested split export |
| [docs/eda/isic2024_tabular_baselines.md](docs/eda/isic2024_tabular_baselines.md) | tabular baseline 실행, missing policy, nested CV summary |
| [docs/plan/2026-05-14_after_5th_meeting/isic2024_strict_input_data_protocol_presentation_20260514.md](docs/plan/2026-05-14_after_5th_meeting/isic2024_strict_input_data_protocol_presentation_20260514.md) | strict input data protocol 발표 요약 |
| [docs/weekly_report/2026-05-14/](docs/weekly_report/2026-05-14/) | 관련 연구와 진행 메모 |

## Version Control

Track:

```text
source code
experiment configs
docs
small evidence files
small result tables under experiments/tables/
```

Do not track:

```text
raw data
processed datasets
split CSV artifacts
checkpoints
MLflow logs
large generated outputs
cache directories
smoke result tables
```

큰 실행 산출물은 `experiments/outputs/`에 두고, Git에는 `experiments/tables/<family>/<run_group_id>/nested_cv/`의 작은 CSV/JSON/Markdown 요약만 올린다.
