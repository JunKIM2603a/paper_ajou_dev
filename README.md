# CBIS-DDSM Model Benchmark

`1st_after` 아래의 각 폴더는 하나의 모델 실험 단위입니다. 모든 모델은 동일한 `CBIS-DDSM: Breast Cancer Image Dataset` 크롭 이미지를 사용해 이진 분류(`BENIGN/BENIGN_WITHOUT_CALLBACK` vs `MALIGNANT`)를 수행하고, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `AUC-ROC`를 `MLflow`에 기록합니다.

## 구성

- `cbis_ddsm_benchmark/`
- `1st_after/<모델명>/config.json`
- `1st_after/<모델명>/run.ps1`
- `run_all_models.py`

## 실행 준비

```powershell
python -m pip install -r requirements.txt
```

## 단일 모델 실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\1st_after\ResNet-50\run.ps1"
```

## 전체 모델 실행

```powershell
python .\run_all_models.py --dataset-root ".\dataset\"
```

## MLflow UI

```powershell
powershell -ExecutionPolicy Bypass -File ".\start_mlflow.ps1"
```

부모 런(`tags.role = model_parent`)은 모델별 최고 결과 비교용이고, 자식 런(`tags.role = hyperparameter_trial`)은 하이퍼파라미터별 상세 결과 확인용입니다.

## 리더보드 CSV 추출

```powershell
python -m cbis_ddsm_benchmark.mlflow_report
```

## HTML 리포트 추출

```powershell
python -m cbis_ddsm_benchmark.mlflow_html_report
```

## 체크포인트 메모

일부 사전학습 모델은 추가 패키지 또는 별도 체크포인트가 필요합니다.

- `BioMedCLIP`, `CheXzero`: `open_clip_torch`
- `MedCLIP`: `transformers`
- `DeiT-S`, `DINOv2 ViT-S`, `RETFound`: `timm`
- `EyePACS`, `HAM10000`, `TorchXRayVision`: 별도 가중치 체크포인트가 있으면 `config.json`의 `checkpoint_path`를 채워 넣으면 됩니다.
## Reset Results

```powershell
powershell -ExecutionPolicy Bypass -File ".\reset_results.ps1"
```

`artifacts\cache`까지 함께 지우려면 아래처럼 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File ".\reset_results.ps1" -ClearCache
```

## Reproducible Seed

전체 실행을 항상 같은 seed로 고정하려면 아래처럼 실행합니다.

```powershell
python .\run_all_models.py --dataset-root ".\dataset\" --seed 42
```
