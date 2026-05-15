# ISIC2024 멀티모달 연구 워크스페이스

이 저장소는 ISIC2024 기반 초희귀 malignant target 멀티모달 학습 연구 코드를 정리하기 위한 작업 공간이다. 현재 확정된 기본 입력 계약은 `image + strict_input`이다. LUPI / privileged supervision 아이디어는 기본 프로젝트가 아니라 `train-only privileged supervision candidate`로만 다룬다.

## 프로젝트 구조

```text
src/isic2024_multimodal/          # Python 패키지
notebooks/isic_2024/              # EDA 및 후속 분석 노트북
experiments/configs/              # 버전 관리되는 실험 config
experiments/evidence/             # 추적 가능한 작은 evidence 표와 메모
experiments/tables/               # 논문/결과용 요약 표
experiments/outputs/              # 생성된 checkpoint, summary, cache
experiments/logs/                 # MLflow FileStore 및 실행 로그
data/raw/isic_2024_challenge/     # 로컬 ISIC2024 raw data, git 추적 제외
data/processed/                   # 생성된 dataset, git 추적 제외
data/splits/                      # 생성된 split 파일, git 추적 제외
docs/                             # 계획, 다이어그램, 회의록, 보고서
```

## 패키지 구조

```text
isic2024_multimodal/
  cli/              # 실행 가능한 entrypoint
  data/             # dataset 경로 해석, image/tabular 로딩, split
  features/         # strict_input feature set과 train-only preprocessing spec
  models/           # image, tabular, fusion, head module
  baselines/        # image/tabular baseline 구현
  training/         # training loop와 재현성 helper
  evaluation/       # metric
  experiments/      # dataset spec, family path, selection registry helper
  reporting/        # MLflow CSV/HTML report
  research/         # 향후 LUPI 변형 등 train-only privileged supervision candidate
  utils/            # config 및 runtime helper
```

## 기본 경로

- Dataset root: `data/raw`
  - 현재 로컬 raw 파일은 `data/raw/train-metadata.csv`, `data/raw/train-image/image/` 형태로 배치되어 있다.
  - 새 raw data를 정리할 때는 `data/raw/isic_2024_challenge/` 아래에 두는 것을 권장한다.
- Image baseline config: `experiments/configs/image_baselines`
- Tabular evidence root: `experiments/evidence/eda/isic_2024`
- Output: `experiments/outputs`
- Table: `experiments/tables`
- MLflow FileStore: `experiments/logs/mlruns`
- MLflow SQLite DB 사용 시: `experiments/logs/mlflow.db`

## 논문 실험 운영 구조

Baseline과 final model 실험은 family 단위로 독립 관리한다. `tabular_baselines`와 `image_baselines`는 서로 순서 의존이 없는 단위 시험이고, `multimodal_baselines`는 tabular/image selection registry를 참조해 조합 후보를 만들 수 있다. `final_paper_model`은 multimodal baseline selection을 기반으로 ablation을 진행한다.

```text
experiments/configs/suites/              # family별 실행 suite
experiments/configs/dataset_specs/       # versioned dataset input contract
experiments/registry/models/             # 후보군 registry
experiments/registry/selections/         # run_group별 best model reference
experiments/outputs/<family>/<run_id>/   # family별 생성 결과
experiments/tables/<family>/<run_id>/    # family별 leaderboard/report
data/processed/datasets/<dataset_id>/    # raw에서 파생된 versioned dataset
```

공통 family runner는 운영 메타데이터를 먼저 남긴 뒤 기존 runner를 subprocess로 호출한다. 결과가 성공하면 local `summary.json`을 다시 훑어 family별 selection registry와 local leaderboard를 갱신한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_experiment_family \
  --family tabular_baselines \
  --config experiments/configs/suites/tabular_baselines.json \
  --run-group-id tabular_strict_v1_gpu0 \
  --devices 0
```

주요 공통 옵션은 다음과 같다.

- `--dataset-spec`: `experiments/configs/dataset_specs/*.json` override
- `--run-group-id`: 같은 실험 묶음을 재실행/조회하기 위한 id
- `--smoke`: suite의 smoke cap을 적용하고 `experiments/outputs/smoke/<run_id>/<family>`에 저장
- `--preflight-only`: dataset spec, split, output/table 경로 manifest만 확인
- `--reset-family-output`: 해당 family/run group의 output/table만 삭제
- `--skip-reports`: MLflow CSV/HTML report 생성을 생략

Family reset은 다른 family, raw data, split, registry를 건드리지 않는다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_experiment_family \
  --family tabular_baselines \
  --run-group-id tabular_strict_v1_gpu0 \
  --reset-family-output
```

## 자주 쓰는 명령

모든 명령은 프로젝트 conda 환경을 사용하고 `PYTHONPATH=./src`를 유지한다. 현재 로컬 환경 이름은 `paper`이다.

공통 실행 형식은 다음과 같다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m <module>
```

- `conda run -n paper`: `paper` conda 환경에서 실행한다.
- `ISIC2024_EXPECTED_CONDA_ENV=paper`: 일부 CLI의 conda 환경 검사를 현재 로컬 env 이름에 맞춘다.
- `env PYTHONPATH=./src`: `src/` 아래의 `isic2024_multimodal` 패키지를 import할 수 있게 한다.
- `python -m isic2024_multimodal.cli.<command>`: `src/isic2024_multimodal/cli/` 아래의 CLI entrypoint를 실행한다.

### 1. Strict input dataset과 patient-level Nested CV split 생성

Strict input dataset, train-only `iddx_full` sidecar, patient-level Triple Stratified Nested CV artifact를 생성한다. 자세한 설명은 `docs/eda/isic2024_strict_input_export.md`와 `docs/reproducibility.md`에 있다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw
```

생성되는 주요 파일은 다음과 같다.

- `data/processed/isic2024_strict_model_input.csv`
  - ordinary inference-time tabular feature만 포함한다.
  - `iddx_full`, diagnosis text, pathology-derived field는 포함하지 않는다.
- `data/processed/isic2024_iddx_full_train_only_sidecar.csv`
  - `iddx_full`을 `iddx_full_train_only`로 분리한 sidecar 파일이다.
  - LUPI / privileged supervision candidate에서 training-only signal로만 사용해야 한다.
- `data/splits/isic2024_official_train_nested_5x4_seed42.csv`
  - outer 5-fold `cv_test_fold` / `outer_test`와 각 `cv_train` 내부 inner 4-fold `inner_train` / `inner_validation` assignment다.
  - 모든 split role은 patient-level이며 같은 Triple Stratified objective로 만든다.
- `experiments/evidence/validation_protocol/isic2024_strict_input_export_summary_seed42.json`
  - column contract, nested split count, patient overlap audit, outer/inner balance audit 요약이다.

기본값을 바꿔야 할 때 사용할 수 있는 주요 옵션은 다음과 같다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw \
  --seed 42 \
  --outer-folds 5 \
  --inner-folds 4
```

주의: `data/raw/isic_2024_challenge/`는 읽기 전용 raw data 영역이다. 이 명령은 raw data를 수정하지 않고 `data/processed/`, `data/splits/`, `experiments/evidence/`에 파생 artifact를 쓴다.

### 2. Strict input export 테스트

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py
```

이 테스트는 strict input export 로직이 기본 contract를 지키는지 확인한다. 특히 ordinary input에 `iddx_full` 같은 privileged field가 섞이지 않았는지, patient-level split artifact가 기대 구조를 갖는지 확인하는 용도다.

### 3. Tabular baseline 실행

Tabular baseline은 nested split CSV와 inner-validation-selected threshold를 사용한다. 자세한 설명은 `docs/eda/isic2024_tabular_baselines.md`에 있다.

GPU 사용 전에는 `paper` 환경에서 CUDA가 보이는지 확인한다.

```bash
conda run -n paper python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

`torch.cuda.is_available()`가 `True`이고 GPU 수가 1 이상이어야 tabular GPU 실행이 가능하다. 현재 GPU 사용을 위해 확인한 기준 환경은 `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `torchaudio 2.5.1+cu121`, `lightgbm 4.6.0`이다.

GPU를 사용할 수 있으면 GPU-capable tabular 모델은 기본적으로 모델별 subprocess runner로 실행한다. 단일 GPU에서도 `run_all_tabular_models --devices 0`를 권장한다. 모델별 프로세스가 분리되어 FT-Transformer와 tree model 사이의 GPU 메모리 잔류 위험이 낮다.

권장 초기화는 위의 `run_experiment_family --reset-family-output`이다. 기존 직접 runner 결과와 local MLflow history까지 모두 지우는 전체 실험 로그 초기화가 필요할 때만 아래 명령을 쓴다.

```bash
rm -rf experiments/outputs/tabular_baselines \
       experiments/outputs/tabular_baselines_smoke \
       experiments/logs/mlruns \
       experiments/logs/mlflow.db
```

이 명령도 `data/raw`, `data/processed`, `data/splits`, `experiments/registry`는 삭제하지 않는다. split CSV는 paper-valid protocol의 입력이므로, split을 의도적으로 다시 만들 때가 아니면 유지한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0
```

이 명령은 GPU-capable tabular-only baseline을 실행한다.

- `--models`: 실행할 tabular model 목록이다.
- `--feature-sets strict_main_input`: strict inference-time input feature set만 사용한다.
- `--devices 0`: 모델별 subprocess에 GPU를 배정한다.
- LightGBM은 WSL/CUDA 환경에서 OpenCL GPU를 요구하지 않도록 CPU backend로 실행된다. XGBoost, CatBoost, FT-Transformer 계열은 CUDA를 사용한다.
- 기본 split 파일:
  - `data/splits/isic2024_official_train_nested_5x4_seed42.csv`
- 기본 실행은 `--outer-fold 0 --inner-fold 0` 조합을 읽는다. 논문 결과는 outer fold별로 반복해 fold-wise summary를 만든다.
- threshold는 `inner_validation`에서 F1 기준으로 선택한다.
- `outer_test`는 threshold 선택, feature selection, preprocessing fitting, model choice에 사용하면 안 된다.

실행 로그는 `[YYYY-MM-DD HH:MM:SS]` prefix로 preflight, model subprocess, report 생성의 시작/종료와 duration을 찍는다. 각 model subprocess 내부에서는 data/protocol load, trial, final_test 시작/종료 시간이 남고, 각 `summary.json`에는 `started_at`, `ended_at`, `duration_seconds`, `timing_seconds`가 저장된다.

`run_all_tabular_models`는 기본적으로 timestamp 기반 `run_group_id`를 만들고, 실행 후 CSV/HTML report를 해당 run group으로 필터링한다. 같은 결과 묶음을 명시적으로 재생산하거나 report를 다시 만들고 싶으면 `--run-group-id <id>`를 직접 지정한다.

`logistic_regression`, `svm`, `mlp`는 CPU 실행 시 sklearn estimator를 쓰지만 `--device cuda`를 주면 repo-native torch estimator로 바뀐다. Paper-facing sklearn baseline으로 비교하려면 이 세 모델은 CPU 명령으로 따로 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --dataset-root data/raw \
  --models logistic_regression svm mlp \
  --feature-sets strict_main_input
```

빠른 smoke test는 모델과 row 수를 줄여서 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost \
  --feature-sets strict_main_input \
  --devices 0 \
  --max-train-rows 1000 \
  --max-val-rows 500 \
  --max-test-rows 500 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports
```

옵션을 생략하면 기본값이 `--device cpu`라서 GPU를 사용하지 않는다.
단일 모델 디버깅이 필요할 때만 `run_tabular_baseline --device cuda`를 직접 사용한다.

### 4. Image baseline 단일 config 실행

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --dataset-root data/raw \
  --config experiments/configs/image_baselines/resnet50/config.json
```

이 명령은 지정한 image baseline config 하나를 실행한다.

- `--config`: image model config JSON 경로다.
- 기본 dataset root는 `data/raw/isic_2024_challenge`다.
- 기본 output root는 `experiments/outputs/image_baselines`다.
- split은 tabular baseline과 같은 nested split CSV를 읽는다. image manifest는 `isic_id`로 이 artifact에 join되며, patient overlap audit이 0이어야 실행된다.

빠른 동작 확인은 epoch, trial, sample 수를 줄여서 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --dataset-root data/raw \
  --config experiments/configs/image_baselines/resnet50/config.json \
  --max-trials 1 \
  --epochs-override 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128
```

사전학습 가중치나 외부 model hub 접근 없이 smoke test가 필요하면 다음 옵션을 추가한다.

```bash
--disable-pretrained
```

### 5. 모든 image baseline 실행

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_image_models \
  --dataset-root data/raw
```

`experiments/configs/image_baselines/*/config.json`을 찾아 순차 실행한다. 실행 후 기본적으로 MLflow CSV/HTML report도 생성한다.

자주 쓰는 옵션은 다음과 같다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_image_models \
  --dataset-root data/raw \
  --models resnet50 \
  --max-trials 1 \
  --epochs-override 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --max-test-samples 128
```

GPU를 여러 개 사용할 수 있으면 다음처럼 병렬 실행할 수 있다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_image_models \
  --dataset-root data/raw \
  --devices 0 1
```

### 6. 모든 tabular model 실행

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost catboost lightgbm ft_transformer ft_transformer_external \
  --feature-sets strict_main_input \
  --devices 0
```

GPU-capable tabular model 목록을 모델 단위 subprocess로 실행하고, 실행 후 report를 생성한다. `--devices 0`은 각 subprocess에 `CUDA_VISIBLE_DEVICES=0`을 배정하고 `--device cuda`를 전달한다. GPU가 보이지 않거나 PyTorch CUDA 초기화가 실패하면 preflight 단계에서 중단된다.

기존 직접 runner 결과와 local MLflow history를 함께 비우는 전체 로그 초기화가 필요할 때만 다음 명령을 쓴다. Family 단위 초기화는 `run_experiment_family --reset-family-output`을 우선 사용한다.

```bash
rm -rf experiments/outputs/tabular_baselines \
       experiments/outputs/tabular_baselines_smoke \
       experiments/logs/mlruns \
       experiments/logs/mlflow.db
```

`data/raw`, `data/processed`, `data/splits`, `experiments/registry`는 이 초기화 대상이 아니다.

실행 중에는 `run_all_tabular_models`가 preflight, 각 모델 subprocess, report 생성의 시작/종료 시각과 duration을 출력한다. `run_tabular_baseline`이 남기는 각 `summary.json`의 `timing_seconds`에서 `prepare_splits_seconds`, `build_estimator_seconds`, `fit_seconds`, `select_threshold_seconds`, `evaluate_train_seconds`, `evaluate_val_seconds`, `evaluate_test_seconds`를 확인할 수 있다.

CPU sklearn baseline을 전체 runner로 실행하려면 GPU 옵션 없이 별도로 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models logistic_regression svm mlp \
  --feature-sets strict_main_input
```

빠른 GPU smoke test는 모델과 row 수를 줄여서 실행한다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost \
  --feature-sets strict_main_input \
  --max-train-rows 1000 \
  --max-val-rows 500 \
  --max-test-rows 500 \
  --devices 0 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports
```

### 7. MLflow report 생성

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.generate_reports \
  --experiment-name ISIC2024-Tabular-Baselines \
  --output-prefix experiments/tables/tabular_mlflow_report
```

이 명령은 MLflow에 기록된 run을 CSV와 HTML로 정리한다.

생성 파일은 다음과 같다.

- `experiments/tables/tabular_mlflow_report.csv`
- `experiments/tables/tabular_mlflow_report.html`

기본 정렬 기준은 `best_pauc_above_tpr80`이다. ultra-rare malignant target에서는 accuracy 단독 보고가 아니라 pAUC, AUC, F1, precision, recall, balanced accuracy를 함께 확인해야 한다.

### 8. Multimodal entrypoint 확인

멀티모달 entrypoint는 향후 `image + strict_input` fusion 실험의 통합 지점으로 준비되어 있다.

```bash
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_multimodal_experiment --help
```

`--help`는 실제 학습을 실행하지 않고 사용 가능한 옵션만 출력한다.

### 권장 실행 순서

처음 환경을 준비하거나 새 split으로 실험을 시작할 때는 다음 순서를 권장한다.

```bash
# 1. strict input과 patient-level split 생성
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset \
  --dataset-root data/raw

# 2. export contract 테스트
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py

# 3. 빠른 tabular GPU smoke test
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models \
  --dataset-root data/raw \
  --models xgboost \
  --feature-sets strict_main_input \
  --devices 0 \
  --max-train-rows 1000 \
  --max-val-rows 500 \
  --max-test-rows 500 \
  --output-root experiments/outputs/tabular_baselines_smoke \
  --skip-reports

# 4. 결과 report 생성
conda run -n paper env ISIC2024_EXPECTED_CONDA_ENV=paper PYTHONPATH=./src python -m isic2024_multimodal.cli.generate_reports \
  --experiment-name ISIC2024-Tabular-Baselines \
  --output-prefix experiments/tables/tabular_mlflow_report
```

### 실험 안전 원칙

- paper-facing 실험은 patient-level split을 사용해야 한다.
- preprocessing, imputation, scaling, encoding, feature selection, class weight, sampler는 training fold에서만 fit해야 한다.
- threshold는 validation set에서만 선택해야 한다.
- `iddx_full`과 diagnosis text는 ordinary inference-time input이 아니다.
- LUPI / privileged supervision은 `train-only privileged supervision candidate`로 명시된 실험에서만 사용한다.
- test fold는 최종 보고용이며 model selection, threshold selection, calibration fitting에 사용하지 않는다.

## 버전 관리 원칙

source code, config, 문서, 작은 evidence table은 추적한다. raw data, 생성된 dataset, checkpoint, cache, MLflow log, 큰 렌더링 파일은 추적하지 않는다. 새 파일과 디렉터리 이름은 외부 논문/출처 제목을 보존해야 하는 경우가 아니라면 English `snake_case`를 사용한다.
