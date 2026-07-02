# PanDerm 모델 리뷰 — 사용 가이드 & 파이프라인

생성일: 2026-07-02 · 최종 개정: 2026-07-02 (실무 사용 가이드 중심 재구성)

> **논문**: [A Multimodal Vision Foundation Model for Clinical Dermatology](https://www.nature.com/articles/s41591-025-03747-y) — *Nature Medicine* 31, 2691–2702 (2025)
> **Arxiv**: https://arxiv.org/pdf/2410.15038
> **공식 코드**: https://github.com/SiyuanYan1/PanDerm · 본 저장소 사본: `PanDerm/`
> **라이선스**: CC-BY-NC-ND 4.0 (비상업적 학술 연구 목적에 한함)

본 문서는 **"이 문서 하나로 PanDerm을 우리 데이터셋(aptos2019, Oral_Diseases)에 적용"** 하는 것을 목표로,
*실무 사용 가이드 → 모델/내부 동작 → 우리 연구 연관성* 순으로 구성한다.

---

## 1. 개요

기존 피부과 AI는 단일 태스크(예: dermoscopy 피부암 진단)에만 특화되어 다양한 임상 시나리오에 적용하기 어려웠다.
PanDerm은 이를 해결하기 위한 **4가지 영상 모달리티 통합 범용 피부과 Foundation Model**이다.

- **사전학습 데이터**: 11개 임상기관, 4개 모달리티에 걸친 **200만 장 이상**의 비레이블 피부 영상으로 self-supervised 사전학습
- **평가**: 28개 벤치마크(피부암 스크리닝, 감별 진단, 병변 분할, longitudinal 모니터링, 전이/예후 예측 등) 전부 SOTA
- **데이터 효율**: labeled data **10%** 만으로 기존 모델 전체 수준 이상

**Reader study 핵심 결과 (논문 Abstract 확인)**

| 항목 | 결과 |
|---|---|
| 초기 흑색종 검출 (longitudinal) | 임상의 대비 **+10.2%** |
| Dermoscopy 피부암 진단 정확도 | 임상의 보조 시 **+11%** |
| 128개 피부질환 감별 (clinical photo) | 비전문 의료인 **+16.5%** |

> **사용 전 필수 이해**: PanDerm은 범용 모델이므로 **반드시 Linear Probing 또는 Fine-tuning** 을 거쳐
> 특정 태스크에 적용해야 한다 (README 명시).

---

## 2. 모델 종류 및 가중치

| 모델 | 아키텍처 | Embed dim / Depth / Heads | 체크포인트 파일 | 비고 |
|---|---|---|---|---|
| **PanDerm (Large)** | ViT-L/16 | 1024 / 24 / 16 | `panderm_ll_data6_checkpoint-499.pth` | 논문 제안 모델 (`PanDerm_Large_LP` / `PanDerm_Large_FT`) |
| **PanDerm Base** | ViT-B/16 | 768 / 12 / 12 | `panderm_bb_data6_checkpoint-499.pth` | 경량 버전 (`PanDerm_Base_LP` / `PanDerm_Base_FT`) |
| **DermLIP_PanDerm** | ViT-B/16 | — | HuggingFace `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | CLIP 계열, 텍스트 정렬 |

- 입력 해상도 224×224, Patch 16×16 (공통)
- 가중치 다운로드 링크: [README §1](../../PanDerm/README.md) (Google Drive / HuggingFace)
- 본 저장소 체크포인트 메타: [`PanDerm/checkpoint/info.txt`](../../PanDerm/checkpoint/info.txt)
- **비교 평가용 baseline**: `get_encoder`에는 PanDerm 외 `SwAVDerm`, `dinov2`, `imgnet_large21k` 도 선택 가능
  ([`builder.py::get_encoder`](../../PanDerm/classification/models/builder.py#L47))

---

## 3. 환경 설치

분류(classification)와 분할(segmentation)은 **의존성이 달라 conda env를 분리**한다.

**Classification (Linear Eval / Fine-tuning)**
```bash
conda create -n PanDerm python=3.10 -y && conda activate PanDerm
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
cd PanDerm/classification && pip install -r requirements.txt
```

**Segmentation (MMSegmentation 기반)**
```bash
conda create -n dermseg python=3.10 -y && conda activate dermseg
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
cd PanDerm/segmentation && pip install -r requirements.txt
pip install -U openmim && mim install mmengine==0.10.4 mmcv==2.1.0 mmsegmentation==1.2.2
```

---

## 4. 데이터 준비

학습/평가에 쓸 데이터는 **CSV 한 장**으로 정의한다.

**필수 컬럼**

| 컬럼 | 설명 |
|---|---|
| `image` | 이미지 파일명 (예: `ISIC_0034524.jpg`) — `root_path`와 결합해 경로 생성 |
| `split` | `train` / `val` / `test` |
| `label` | 다중 분류 시 정수 클래스 (0,1,2,…) |
| `binary_label` | 이진 분류 시 0/1 (`--nb_classes 2` 일 때 사용) |

```csv
image,label,split
ISIC_0034524.jpg,1,train
ISIC_0034526.jpg,4,val
ISIC_0034527.jpg,3,test
```

**우리 데이터셋 매핑** (`PanDerm/data/.../Linear Evaluation/`)

| 데이터셋 | 클래스 수 | label_map | 클래스 |
|---|---:|---|---|
| aptos2019 | 5 (`--nb_classes 5`) | [`aptos2019_label_map.csv`](../../PanDerm/data/aptos2019/Linear%20Evaluation/aptos2019_label_map.csv) | no_dr / mild / moderate / severe / proliferative_dr |
| Oral_Diseases | 7 (`--nb_classes 7`) | [`oral_diseases_label_map.csv`](../../PanDerm/data/Oral_Diseases/Linear%20Evaluation/oral_diseases_label_map.csv) | CaS / CoS / Gum / MC / OC / OLP / OT |

> **주의 (8차 미팅 지시)**: `Nail Disease`, `Metastatic_Tissue` 는 PanDerm 사전학습/평가에 사용된 데이터셋이므로 **본 실험에서 배제**.

---

## 5. 사용 파이프라인 (실행 가이드)

### 5.1 Feature Extraction & UMAP

frozen encoder로 임베딩만 뽑아 시각화·탐색하는 가장 가벼운 사용법.
- 노트북: [`classification/feature_extraction_and_umap.ipynb`](../../PanDerm/classification/feature_extraction_and_umap.ipynb)
- 용도: representation 품질 사전 점검, 클래스 군집/아웃라이어 탐색

### 5.2 Linear Evaluation (frozen encoder + 선형 분류기)

> **내부 동작 상세**: [linear_evaluation_method_explained.md](../../PanDerm/Linear%20Evaluation/linear_evaluation_method_explained.md)

**실행**
```bash
cd PanDerm/classification
CUDA_VISIBLE_DEVICES=0 python3 linear_eval.py \
  --model "PanDerm_Large_LP" \
  --nb_classes 5 \                 # aptos2019: 5, Oral_Diseases: 7
  --percent_data 1.0 \             # 0.1 → label efficiency 실험
  --batch_size 1000 \
  --csv_filename "result.csv" \
  --output_dir "/path/to/output_dir/" \
  --csv_path  "/path/to/dataset.csv" \
  --root_path "/path/to/images/" \
  --pretrained_checkpoint "/path/to/panderm_ll_data6_checkpoint-499.pth"
```
- 9개 공개 데이터셋 재현: `bash script/lp_reproduce.sh` (모델·데이터셋별 커맨드는 스크립트 내 주석 참고)

**핵심 흐름**
```
CSV → Derm_Dataset → DataLoader → PanDerm Encoder(frozen) → 1024-d feature
  → LogisticRegression(torch.nn.Linear + LBFGS) → 지표 계산 → 결과 CSV + Confusion Matrix
```

**동작상 주의점 (코드 대조 확인)**
- 추론 전처리: `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)`
  ([`builder.py::get_eval_transforms`](../../PanDerm/classification/models/builder.py#L24))
- 손실: `CrossEntropyLoss(weight=None)` → **불균형 데이터셋에서 다수 클래스로 편향**
- 정규화 강도 `C = (feat_dim × NUM_C) / 100`, 여기서 `NUM_C`는 `--nb_classes`가 아니라
  **train 라벨에 실제 등장한 distinct 클래스 수** (탐색 없이 자동 설정)
  ([`linear_probe.py::train_linear_probe`](../../PanDerm/classification/panderm_model/downstream/eval_features/linear_probe.py#L113))
- `linear_eval.py`는 `valid_feats=None`으로 호출 → val 피처는 추출되지만 **학습에 미사용**
  (train-only fit, [`linear_probe.py` L121](../../PanDerm/classification/panderm_model/downstream/eval_features/linear_probe.py#L121))
- 다중 분류 AUROC는 OvO Macro, AUPR은 OvR Macro로 계산 → [세부 분석](panderm_linear_eval_analysis.md)

### 5.3 Fine-tuning (backbone 업데이트)

> **Entry**: [`run_class_finetuning.py`](../../PanDerm/classification/run_class_finetuning.py) ·
> **Engine**: [`engine_for_finetuning.py`](../../PanDerm/classification/furnace/engine_for_finetuning.py) ·
> **스크립트**: [`script/finetune_train.sh`](../../PanDerm/classification/script/finetune_train.sh) / [`finetune_test.sh`](../../PanDerm/classification/script/finetune_test.sh)

**권장 설정 (README)**: batch 128, lr 5e-4, epochs 50, WeightedRandomSampler + TTA 활성화 (데이터셋 간 robust)

**학습 — 불균형 데이터셋(aptos2019 / Oral_Diseases) 권장 커맨드**
```bash
cd PanDerm/classification
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
  --model PanDerm_Large_FT \
  --pretrained_checkpoint /path/to/panderm_ll_data6_checkpoint-499.pth \
  --nb_classes 5 \                 # aptos2019: 5, Oral_Diseases: 7
  --batch_size 128 --lr 5e-4 --update_freq 1 \
  --warmup_epochs 10 --epochs 50 \
  --layer_decay 0.65 --drop_path 0.2 --weight_decay 0.05 \
  --mixup 0.8 --cutmix 1.0 \
  --weights \                      # WeightedRandomSampler (불균형 대응)
  --monitor recall \               # 불균형 시 Balanced Recall 기준 best checkpoint (기본 acc)
  --sin_pos_emb --no_auto_resume --imagenet_default_mean_and_std \
  --output_dir /path/to/FT_Res/ \
  --csv_path /path/to/dataset.csv --root_path /path/to/images/ --seed 0
```

**평가 / 추론 (학습된 checkpoint 사용)**
```bash
python3 run_class_finetuning.py ... \
  --resume /path/to/checkpoint-best.pth \
  --eval \                         # 추론 모드
  --TTA                            # 5-augmentation 앙상블 추론
```
- 스크립트: `bash script/finetune_test.sh` (`--resume` 경로만 수정)
- `layer_decay` 파서 기본값은 0.9이나 모든 스크립트가 **0.65** 명시 사용 → 0.65 권장

### 5.4 Segmentation (병변 분할)

> 상세: [`Segmentation.md`](../../PanDerm/Segmentation.md) · Entry [`segmentation/run.py`](../../PanDerm/segmentation/run.py)

```bash
conda activate dermseg && cd PanDerm/segmentation
# run.sh 안의 --parent_path / --dataset(ISIC2018, HAM10000) 수정 후
bash run.sh            # 학습 (epoch 100, lr 1e-4)
# 평가 시: run.sh에 --evaluate 추가
```
- 아키텍처: CAE Backbone(ViT-L, multi-scale layer [7,11,15,23]) → **UPerHead** → bilinear upsample → 2-class(병변/배경)
- 가중치 로딩: `encoder.*` 키에서 `encoder.` prefix 제거 후 backbone에 `strict=False` 로드
- 자체 이미지 추론 스타터: [`segmentation/evaluate.ipynb`](../../PanDerm/segmentation/evaluate.ipynb)

---

## 6. 모델 아키텍처 & 사전학습

**사전학습 데이터 구성** (총 2,149,706장, 모두 unlabeled)

| 모달리티 | 이미지 수 | 비율 |
|---|---:|---:|
| TBP (Total Body Photography) 타일 | 757,890 | 35.3% |
| Dermatopathology 타일 | 537,047 | 25.0% |
| Clinical 이미지 | 460,328 | 21.4% |
| Dermoscopy 이미지 | 384,441 | 17.9% |

**사전학습 메커니즘: CAEv2 + CLIP 정렬 (Masked Latent Alignment)**
```
입력 이미지
  ├── 50% masking
  │     ├── Visible patches → ViT-L Encoder → visible latent
  │     └── Mask tokens → Cross-attention Regressor → masked latent 예측
  └── CLIP-Large Teacher → visible/masked patch supervision (Latent Alignment Loss)
```
- Decoder 없이 Regressor만으로 masked patch 예측 (효율적) · CLIP latent supervision으로 의미론적 표현 학습
- ImageNet-1K 초기화 후 피부과 데이터로 continue pretraining
- 설정: AdamW, lr 1.5e-3, batch 1920, 500 epochs(warmup 20), 4× H100 80GB ≈ 5일

---

## 7. 파이프라인 비교

| 구분 | Linear Evaluation | Fine-tuning | Segmentation |
|---|---|---|---|
| 목적 | 표현(representation) 품질 평가 | downstream 태스크 성능 최적화 | 병변 분할 (픽셀 단위 병변/배경 분리) |
| Encoder 업데이트 | ❌ 고정(frozen) | ✅ backbone 학습 | ✅ backbone 학습 |
| 학습 대상 | `nn.Linear` 하나 | backbone + head | backbone + UPerHead |
| Optimizer | LBFGS | AdamW (layer_decay 0.65) | AdamW |
| 데이터 증강 | Resize+CenterCrop만 | RandAug, Mixup(0.8), CutMix(1.0) | — |
| 불균형 대응 | ❌ 없음 | ✅ `--weights` (WeightedRandomSampler) | 해당 없음 |
| 추론 | 단일 forward | `--TTA` (5-aug 앙상블) | — |
| 계산 비용 | 매우 낮음 | 보통 (50 epochs) | 보통 (100 epochs) |
| Entry | `linear_eval.py` | `run_class_finetuning.py` | `segmentation/run.py` |

---

## 8. 우리 연구와의 연관성

### 8-1) 집중 데이터셋 Linear Eval 성능 특성

| 데이터셋 | 태스크 | Accuracy | Balanced Acc | 주요 관찰 |
|---|---|---:|---:|---|
| aptos2019 | 다중 분류(5) | 0.814 | 0.628 | Accuracy–BACC 갭 +0.186 — 다수 클래스(No DR) 편향 심각 |
| Oral_Diseases | 다중 분류(7) | 0.778 | 0.811 | OLP Recall 0.30 — 집중 오분류 패턴 존재 |

> **세부 분석**: [panderm_linear_eval_analysis.md](panderm_linear_eval_analysis.md)

### 8-2) 논문이 명시한 한계와의 대조

| 논문 한계 | 우리 연구 연관성 |
|---|---|
| 피부 질환 중심 커버리지 | Oral_Diseases(구강 점막)는 피부과 외 영역 — 표현 학습 미흡 가능성 |
| 단일 모달리티 평가 중심 | aptos2019(안저)도 피부과 비전통 도메인 — domain gap |
| 소수 클래스 공정성 평가 미흡 | aptos2019 Severe 클래스 AUPR 0.34 — 논문 한계와 직접 대응 |

### 8-3) Fine-tuning 시도 시 기대 효과

| 문제 | Linear Eval 한계 | Fine-tuning 대응 |
|---|---|---|
| aptos2019 소수 클래스 편향 | `CrossEntropyLoss(weight=None)` | `--weights` + `--monitor recall` |
| Oral_Diseases OLP 오분류 | backbone 고정으로 표현력 제한 | backbone 업데이트 + domain adaptation |
| 전반적 성능 향상 | 선형 분류기 표현력 한계 | `--TTA` + backbone fine-tune |

---

## 9. 참고 링크

- 공식 저장소: https://github.com/SiyuanYan1/PanDerm · 후속 모델 DermFM-Zero(PanDerm-2): https://github.com/SiyuanYan1/DermFM-Zero
- Linear Evaluation 동작 원리: [linear_evaluation_method_explained.md](../../PanDerm/Linear%20Evaluation/linear_evaluation_method_explained.md)
- Linear Evaluation 결과/지표 심층 분석: [panderm_linear_eval_analysis.md](panderm_linear_eval_analysis.md)
- 코드 진입점: [`linear_eval.py`](../../PanDerm/classification/linear_eval.py) · [`run_class_finetuning.py`](../../PanDerm/classification/run_class_finetuning.py) · [`segmentation/run.py`](../../PanDerm/segmentation/run.py)
</content>
</invoke>
