# 17. Follow-up Validation Protocol

## 1. Goal

`11~16장`에서 확정한 `strict_base`, `strict_fe`, `strict_main_input`, 그리고 `auxiliary oracle supervision` 설계를 본선 EDA 흐름과 분리해서 재검증할 때의 공통 baseline 규약을 정리한다. 현재 baseline 실행의 tabular 축은 세 feature set을 함께 돌려 상대 기여를 비교하는 방식으로 두고, `relaxed`는 필요할 때만 보조 비교로 남겨 둔다. 이 문서는 `src/eda/isic2024_eda_20260411.ipynb`의 `17. 후속 검증 notebook 안내`를 보강하는 기준 문서다.

실행 코드와 분석 notebook의 역할 분리는 [followup_py_ipynb_split_strategy_17.md](/home/junkim2603a/proj/paper_ajou_dev/docs/eda/isic2024/followup_py_ipynb_split_strategy_17.md)에서 따로 정리한다.

## 2. Primary Metric

후속 baseline 비교의 주 평가지표는 `TPR >= 0.80` 구간의 `pAUC`로 둔다. 코드에서는 `pauc_above_tpr80`라는 이름으로 기록한다.

\[
\mathrm{pAUC}_{\mathrm{TPR}\ge 0.80}
= \int_{0}^{1-0.80} \mathrm{TNR}(\mathrm{FNR}) \, d\mathrm{FNR}
= \int_{0}^{0.20} \mathrm{TNR}(\mathrm{FNR}) \, d\mathrm{FNR}.
\]

구현은 `2024 ISIC Challenge`의 official pAUC 아이디어와 같은 방식으로, label/score를 뒤집은 ROC에서 `max_fpr = 1 - min_tpr`까지만 적분하는 형태를 따른다. 이번 follow-up benchmark에서는 `min_tpr = 0.80`을 사용한다.

하이퍼파라미터 trial 선택은 `validation pauc_above_tpr80` 기준으로 수행하고, parent run leaderboard는 그 child의 `test pauc_above_tpr80`로 비교한다.

## 3. Full Fine-Tuning Definition

image baseline의 `full fine-tuning`은 backbone 전체와 새로 붙인 2-class classifier head를 모두 업데이트하는 설정을 뜻한다. 현재 `src/isic2024_benchmark/trainer.py`는 `optimizer(model.parameters())`를 사용하고, 별도 freezing 로직이 없으므로 runnable한 image config는 모두 full fine-tuning으로 학습된다.

## 4. Baseline Pool

### 4.1 Tabular models

- `xgboost`
- `catboost`
- `svm`
- `logistic_regression`
- `mlp`

Tabular baseline feature set은 아래 세 가지를 기본 비교군으로 둔다.

- `strict_base`: 전처리된 base metadata만 사용
- `strict_fe`: 최종 선택 engineered feature만 사용
- `strict_main_input`: `strict_base + strict_fe`

즉 baseline 질문은 "base만으로 어느 정도 되는가", "engineered feature만으로 어느 정도 되는가", "둘을 합쳤을 때 실제 메인 입력이 얼마나 이득이 있는가"를 함께 보는 구조다.

tabular GPU backend는 다음처럼 둔다.

- `xgboost`, `catboost`: native GPU backend
- `logistic_regression`, `svm`, `mlp`: PyTorch GPU backend

즉 `cuda` 실행 시 모델 family는 유지하되, GPU 친화적인 학습 backend로 전환한다.

### 4.2 Image models

| Model | Current status | Weight source / note | Extra requirement |
|---|---|---|---|
| `BioMedCLIP` | Ready | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 없음 |
| `CheXzero` | Ready | local checkpoint exists in `checkpoints/CheXzero/...` | 없음 |
| `DeiT-S` | Ready | `timm` pretrained | 없음 |
| `DenseNet-121` | Ready | `torchvision DEFAULT` | 없음 |
| `DINOv2 ViT-S` | Ready | `timm` pretrained | 없음 |
| `EfficientNet-B0` | Ready | `torchvision DEFAULT` | 없음 |
| `EyePACS` | Ready | local `eff_net_400x400.pt` matches `torchvision efficientnet_b3`; original `5-class` head is replaced with the benchmark `2-class` head | 없음 |
| `MedCLIP` | Ready | official `medclip` package backend with auto-cached `pytorch_model.bin` under `checkpoints/MedCLIP/official-medclip-vit` | 없음 |
| `MONET` | Ready | Hugging Face `chanwkim/monet` | 없음 |
| `ResNet-50` | Ready | `torchvision DEFAULT` | 없음 |
| `RETFound` | Ready | local `RETFound_cfp_weights.pth` is present and matches the HF loading example | 없음 |
| `TorchXRayVision` | Ready | backend is wired to `xrv.models.DenseNet(weights="densenet121-res224-all")` and caches under `checkpoints/TorchXRayVision` | 없음 |
| `ViT-B_16` | Ready | `torchvision DEFAULT` | 없음 |

이번 follow-up benchmark에서는 `HAM10000`은 제외한다.

즉, 현재 repo 기준으로 바로 full fine-tuning baseline에 올릴 수 있는 쪽은 `BioMedCLIP`, `CheXzero`, `DeiT-S`, `DenseNet-121`, `DINOv2 ViT-S`, `EfficientNet-B0`, `EyePACS`, `MedCLIP`, `MONET`, `ResNet-50`, `RETFound`, `TorchXRayVision`, `ViT-B_16`이다.

## 5. Execution Notes

tabular benchmark, single GPU:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_tabular_baselines --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/final_inputs/feature_sets_recommended.json --feature-sets strict_base strict_fe strict_main_input --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs --device cuda
```

tabular benchmark, 2-GPU parallel:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_all_tabular_models --dataset-root ./dataset/isic-2024-challenge --eda-dir ./artifacts/eda/isic2024 --feature-set-json ./artifacts/eda/isic2024/final_inputs/feature_sets_recommended.json --feature-sets strict_base strict_fe strict_main_input --experiment-name ISIC2024-Tabular-Benchmark --output-root ./artifacts/tabular_runs --devices 0 1
```

image benchmark:

```bash
conda run -n paper_ajou_dev env PYTHONPATH=./src python -m isic2024_benchmark.run_all_models --dataset-root ./dataset/isic-2024-challenge --output-root ./artifacts --experiment-name ISIC2024-Image-Benchmark --seed 42 --devices 0 1 --exclude-models HAM10000
```

`MONET`은 `reference/MONET/README.md`의 Hugging Face 경로를 기준으로 `src/image_baselines/MONET/config.json`에서 바로 호출한다.

image model은 tabular feature set을 직접 입력으로 쓰지 않으므로, 결과 해석에서는 각 image backbone을 `strict_base`, `strict_fe`, `strict_main_input` tabular 기준선과 나란히 놓고 비교한다.
