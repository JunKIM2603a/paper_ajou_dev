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
  reporting/        # MLflow CSV/HTML report
  research/         # 향후 LUPI 변형 등 train-only privileged supervision candidate
  utils/            # config 및 runtime helper
```

## 기본 경로

- Dataset root: `data/raw/isic_2024_challenge`
- Image baseline config: `experiments/configs/image_baselines`
- Tabular evidence root: `experiments/evidence/eda/isic_2024`
- Output: `experiments/outputs`
- Table: `experiments/tables`
- MLflow FileStore: `experiments/logs/mlruns`
- MLflow SQLite DB 사용 시: `experiments/logs/mlflow.db`

## 자주 쓰는 명령

프로젝트 conda 환경을 사용하고 `PYTHONPATH=./src`를 유지한다.

Strict input dataset, train-only `iddx_full` sidecar, patient-level split artifact를 생성한다. 자세한 설명은 `docs/eda/isic2024_strict_input_export.md`에 있다.

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.export_strict_input_dataset
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m pytest tests/test_strict_input_export.py
```

Tabular baseline은 locked split CSV와 validation-selected threshold를 사용한다. 자세한 설명은 `docs/eda/isic2024_tabular_baselines.md`에 있다.

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline \
  --models logistic_regression svm mlp xgboost catboost lightgbm ft_transformer \
  --feature-sets strict_main_input
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_image_baseline \
  --config experiments/configs/image_baselines/resnet50/config.json
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_image_models
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_tabular_baseline
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_all_tabular_models
```

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.generate_reports \
  --experiment-name ISIC2024-Tabular-Baselines \
  --output-prefix experiments/tables/tabular_mlflow_report
```

멀티모달 entrypoint는 향후 통합 지점으로 준비되어 있다.

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_multimodal.cli.run_multimodal_experiment --help
```

## 버전 관리 원칙

source code, config, 문서, 작은 evidence table은 추적한다. raw data, 생성된 dataset, checkpoint, cache, MLflow log, 큰 렌더링 파일은 추적하지 않는다. 새 파일과 디렉터리 이름은 외부 논문/출처 제목을 보존해야 하는 경우가 아니라면 English `snake_case`를 사용한다.
