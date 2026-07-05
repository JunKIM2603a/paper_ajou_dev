# PanDerm 임베딩 활용 — 도메인 일반화 방법론 전략

생성일: 2026-07-05

> **관련 문서**: [`panderm_model_review.md`](panderm_model_review.md) (모델 사용 가이드) · [`panderm_linear_eval_analysis.md`](panderm_linear_eval_analysis.md) (linear-eval 결과 분석)
> **공식 코드**: https://github.com/SiyuanYan1/PanDerm · 본 저장소 사본: `PanDerm/`
> **라이선스**: CC-BY-NC-ND 4.0 (비상업적 학술 연구 목적에 한함)

---

## 0. 목적과 전제

**목표**: PanDerm(피부과 foundation model)의 이미지 임베딩 능력을 활용해, PanDerm이 학습하지 않은 **인접 도메인(구강·점막 질환 등)** 에서 일반화 성능을 확보한다. 필요 시 fine-tuning.

**전제 (연구 조건)**:
- **대상 도메인**: 피부과 인접 (구강/점막) → 도메인 이동이 작아 **특징 재사용성이 높음**.
- **라벨 데이터**: few-shot (클래스당 ~100 이하) → **전체 fine-tuning은 과적합 위험**, frozen/PEFT/probe 계열이 유리.

**현재 파이프라인 (코드 기준)**:
- PanDerm = **ViT-L/16** (embed_dim 1024, depth 24, heads 16), CLS 토큰 + 196 patch 토큰(14×14 grid), 고정 2D sincos pos-embed.
- 지금은 `forward_features(x, is_train=False)`로 **마지막 블록의 CLS 토큰 1024-d만** 추출 → frozen linear probe (LBFGS, 정규화 C 고정, **val 미사용, class weight 없음**). 근거: `PanDerm/classification/models/modeling_finetune.py:496`, `panderm_model/downstream/extract_features.py:32`.
- 중간 레이어 / patch 토큰 / attention은 **API로 노출되지 않음** → forward hook으로 직접 추출 필요.
- 리포에 있으나 **미사용**인 자산: fine-tuning 파이프라인(layer-wise decay, mixup, drop-path, EMA), segmentation용 multi-layer FPN(out_indices [7,11,15,23]), CLIP 변형(DermLIP_PanDerm), `LP_BatchNorm`(feature 보정, `modeling_finetune.py:11`).

**문서 구성**: "PanDerm을 최대한 활용"하는 방법을 **3개 축**(① 어떤 특징을 뽑을까 ② 어떻게 적응/학습할까 ③ 도메인 일반화 특화 기법) + CLIP 활용으로 정리하고, few-shot·인접도메인 시나리오 기준 우선순위와 검증 프로토콜을 제시한다.

---

## 축 A — 어떤 특징을 뽑을 것인가 (Feature Extraction)

> 사용자가 특히 궁금해한 "레이어별 특징 추출/커스텀"이 여기 해당. PanDerm 파라미터는 그대로 두고, **무엇을 읽어낼지**만 바꾸는 저비용 레버.

| 방법 | 무엇을 뽑나 | 언제 유리 | 비용 |
|---|---|---|---|
| A1. 마지막 CLS (현재) | `x[:,0]` (1024) | 기본선 | 무 |
| A2. patch 토큰 풀링 | `x[:,1:]` 평균/최대/attention 풀 (1024) | 국소 패턴·병변 위치 중요할 때 | 무 |
| A3. **중간 레이어 CLS 융합** | 여러 블록의 CLS concat | **OOD/인접도메인에 강함** (중간층이 더 전이 가능) | 낮음 |
| A4. multi-layer patch 피라미드 | out_indices 블록의 patch map (FPN) | segmentation·dense task | 중 |
| A5. CLS ⊕ mean-patch | `[cls, patch.mean]` (2048) | probe 성능 저비용 향상 | 무 |

**핵심 근거(A3)**: ViT의 마지막 층은 사전학습 과제(피부 SSL)에 특화되어 도메인이 바뀌면 오히려 전이력이 떨어질 수 있음. 중간~후반 블록 특징이 더 일반적인 표현을 담는 경우가 많아, **여러 레이어를 concat하거나 최적 레이어를 탐색**하는 것이 인접 도메인에서 저비용·고효율.

**예시 — hook으로 중간 레이어 CLS + patch 풀링 동시 추출** (파라미터 학습 0):
```python
from models import get_encoder
model, eval_transform = get_encoder(args, 'PanDerm_Large_LP')
model.eval().cuda()

layers = [17, 19, 21, 23]          # 0~23 중 원하는 블록 (마지막 4개 예시)
feats = {}
def make_hook(i):
    def hook(_m, _in, out):        # out: (B, 197, 1024) — 최종 LayerNorm 이전
        cls   = out[:, 0]                 # (B,1024)
        patch = out[:, 1:].mean(dim=1)    # (B,1024)  평균 풀
        feats[i] = torch.cat([cls, patch], dim=1)
    return hook
handles = [model.blocks[i].register_forward_hook(make_hook(i)) for i in layers]

with torch.inference_mode():
    _ = model.forward_features(images.cuda(), is_train=False)   # 반환값 대신 hook 결과 사용
emb = torch.cat([feats[i] for i in layers], dim=1)   # (B, 2048*len(layers))
for h in handles: h.remove()
```
- 주의: 블록 출력은 최종 `self.norm`(LayerNorm) 적용 **이전**(`modeling_finetune.py:508-511`). 스케일 편차가 크면 레이어별 LayerNorm 또는 표준화(StandardScaler) 후 concat 권장.
- patch map 필요 시 `out[:,1:].reshape(B,14,14,1024)`. A4 피라미드는 segmentation 백본 `segmentation/models/cae_backbone.py:468`의 out_indices 방식을 참고.
- **레이어 탐색 실험**: 24개 블록 각각의 CLS로 probe를 돌려 어느 층이 인접 도메인에 최적인지 곡선을 그리는 것이 첫 번째 값진 실험(현재 캐시/평가 파이프라인 그대로 재사용 가능).

---

## 축 B — 어떻게 적응/학습할 것인가 (Probing → Fine-tuning 스펙트럼)

> few-shot이므로 **학습 파라미터 수를 통제**하는 것이 핵심. 왼쪽(저파라미터)일수록 과적합에 안전.

| 방법 | 학습 대상 | few-shot 적합 | 비고 |
|---|---|---|---|
| B1. **개선된 linear probe** | 선형층만 | ★★★ | 현재 파이프라인 결함 보완(아래) |
| B2. Prototype/NCM·kNN | 없음(학습 0) | ★★★ | 특징공간 최근접, 초저비용 baseline |
| B3. MLP / attention-pool head | 소형 head | ★★ | 표현력↑, 약간의 과적합 위험 |
| B4. **PEFT** (LoRA/BitFit/LN/VPT) | 백본 일부 소수 | ★★★ | 성능 상한↑ + 과적합 통제, few-shot 최적 |
| B5. 부분 fine-tuning | 마지막 N블록 | ★ | 데이터 늘면 고려, layer-decay 필수 |
| B6. 전체 fine-tuning | 전체 | ✗(few-shot) | 라벨 많을 때만 |

**B1 — 현재 probe의 명백한 결함부터 보완** (거의 무료, 즉효):
- **val 미사용**: `linear_eval.py`가 `valid_feats=None`으로 호출 → val이 fitting에 안 쓰임. few-shot에선 train+val 합치기/교차검증이 중요.
- **class weight 없음** + **정규화 C 고정**(`C=(1024·K)/100`): 불균형·소표본에 취약. → `class_weight='balanced'`, C 그리드 서치.
- **feature 표준화 부재**: 도메인 이동 시 `StandardScaler`(train 통계) 또는 `LP_BatchNorm`(`modeling_finetune.py:11`)으로 특징 보정하면 probe가 안정화.
- 구현 재사용: `panderm_model/downstream/eval_features/linear_probe.py`, `logistic_regression.py`, `metrics.py`.

**B2 — Prototype/NCM (few-shot 표준 baseline, 학습 0)**:
```python
# 클래스별 평균 임베딩(prototype) → 코사인 최근접 분류
proto = {c: F.normalize(train_emb[train_y==c].mean(0), dim=0) for c in classes}
pred  = max(proto, key=lambda c: F.normalize(test_emb,dim=0) @ proto[c])
```
few-shot에서 linear probe보다 강한 경우가 많고, 레이어/풀링 조합 비교의 빠른 잣대로 유용.

**B4 — PEFT (few-shot에 가장 권장하는 "학습" 옵션)**: 백본은 대부분 freeze, 소수 파라미터만 학습 → 성능 상한을 올리면서 과적합을 억제. 모듈명: `model.blocks.{i}.attn.qkv`, `.attn.proj`, `.mlp.fc1/fc2`.
- **LoRA**: attention qkv/proj에 저랭크 어댑터. `peft`의 `LoraConfig(r=8, target_modules=["qkv","proj"])`(suffix 매칭). 학습 파라미터 <1%.
- **BitFit / LayerNorm tuning** (초경량):
  ```python
  for n,p in model.named_parameters():
      p.requires_grad = ('bias' in n) or ('norm' in n)   # bias + LN affine만 학습
  ```
- **VPT (Visual Prompt Tuning)**: 입력 토큰 앞에 학습 가능한 prompt 토큰 prepend, 백본 freeze. 인접 도메인 few-shot에서 강력(구현은 patch_embed 이후 토큰 concat 커스텀 필요).

**B5 — 부분 fine-tuning**: 마지막 N개 블록만 unfreeze + **layer-wise LR decay**. 리포에 이미 존재(`run_class_finetuning.py`, `furnace/optim_factory.py`). 라벨이 few-shot 범위를 벗어날 때만.

---

## 축 C — 도메인 일반화/적응 특화 기법

| 방법 | 라벨 필요 | 핵심 아이디어 | few-shot 적합 |
|---|---|---|---|
| C1. 특징 정규화/보정 | 무 | StandardScaler·LP_BatchNorm·정규화 프리셋 튜닝 | ★★★ |
| C2. 도메인 적응 SSL 사전학습 | 무(비지도) | 대상 도메인 unlabeled 이미지로 MAE/CAE 계속 학습 | 조건부 |
| C3. TTA/TENT | 무 | 테스트시 LN·통계 적응, test-time augmentation | ★★ |
| C4. 특징공간 정렬 | source만 | CORAL/MMD로 derm↔oral 분포 정렬 | ★ |
| C5. 강한 증강 | 라벨셋 | RandAug·color/stain 등 도메인 특화 증강 | ★★ |
| C6. 앙상블/융합 | - | 레이어·정규화 프리셋·모델 앙상블 | ★★ |

- **C1 정규화 프리셋**: `builder.py`가 `imagenet`(주의: std[0]=0.228, 통상 0.229와 다름) / `openai_clip` / `uniform` 3종 제공. 구강 이미지 색분포가 다르면 프리셋 교체만으로 임베딩 품질이 바뀜 → 저비용 실험.
- **C2 도메인 적응 사전학습**: 라벨 없는 구강 이미지가 다수 있으면 MAE/CAE-style로 백본을 대상 도메인에 계속 적응. *단, 이 vendored 리포에는 classification/segmentation만 있고 SSL 사전학습 코드는 없을 가능성 → upstream 사전학습 파이프라인 필요. few-shot·라벨 위주 목표에선 우선순위 낮음.*
- **C3 TENT**: 테스트 배치에서 예측 엔트로피 최소화로 LN affine만 갱신 → 라벨 없이 도메인 이동 완화.

---

## 축 D — 멀티모달/CLIP 활용 (DermLIP)

라벨이 적을 때 강력한 대안. **DermLIP_PanDerm**(HF `redlessone/DermLIP_PanDerm-base-w-PubMed-256`)로:
- **Zero-shot**: 클래스명을 텍스트 프롬프트("a clinical photo of oral lichen planus" 등)로 인코딩 → 이미지 임베딩과 코사인 유사도 분류. 라벨 0개로 baseline 확보.
- **Few-shot(linear probe on CLIP image encoder)**: CLIP 이미지 인코더 특징 + 소수 라벨 probe.
- 주의: 모델 다운로드(인터넷) 필요, ViT-B/16 기반이라 임베딩 차원·전처리가 PanDerm_Large와 다름. 구강 클래스명이 의미론적으로 명확할수록 zero-shot 이득이 큼.

---

## few-shot × 피부과 인접 도메인 — 추천 우선순위

**Tier 1 (즉시·저비용·높은 ROI)**
1. **probe 위생 개선** (B1): val 활용/교차검증, class weight, C 서치, feature 표준화 — 현재 결함 보완.
2. **레이어 탐색 + multi-layer 융합** (A3/A5): 24블록 probe 곡선 → 최적 층/조합 선택. hook만으로 구현, 파라미터 학습 0.
3. **Prototype/NCM·kNN** (B2): few-shot 강baseline, 조합 비교 잣대.
4. **DermLIP zero/few-shot** (D): 라벨 최소로 상한 확인.

**Tier 2 (중간 비용·성능 상한 ↑)**
5. **PEFT — LoRA 또는 BitFit/LN-tuning + linear head** (B4).
6. **VPT** (B4).
7. 정규화 프리셋/증강 튜닝 (C1/C5), 앙상블 (C6).

**Tier 3 (조건부 — 데이터·컴퓨트 필요)**
8. 부분 fine-tuning (B5, layer decay).
9. TTA/TENT (C3), 특징정렬 (C4).
10. 도메인 적응 SSL 사전학습 (C2, unlabeled 구강 이미지 다수 확보 시).

---

## 구현/재사용 포인트

| 목적 | 위치 |
|---|---|
| 모델 로드 + eval transform | `PanDerm/classification/models/builder.py:47` (`get_encoder`) |
| 임베딩 추출 API | `modeling_finetune.py:496` (`forward_features`), `panderm_model/downstream/extract_features.py:32` |
| 블록/attention 구조(hook·LoRA 대상) | `modeling_finetune.py:263`(Block), `attn.qkv/proj`, `mlp.fc1/fc2` |
| 특징 보정 | `LP_BatchNorm` `modeling_finetune.py:11` |
| probe/metrics | `panderm_model/downstream/eval_features/{linear_probe,logistic_regression,metrics}.py` |
| multi-layer 피라미드 참고 | `segmentation/models/cae_backbone.py:468` (out_indices [7,11,15,23]) |
| fine-tuning·layer decay | `run_class_finetuning.py`, `furnace/optim_factory.py` |
| CLIP 변형 | HF `redlessone/DermLIP_PanDerm-base-w-PubMed-256` |

**알려진 주의점**: `linear_type='attentive'` 경로(`modeling_finetune.py:517-521`)는 `AttentiveBlock.forward` 시그니처 불일치로 에러 발생 → attention pooling head가 필요하면 이 부분을 수정/재구현해야 함. 기본 동작 경로는 `'standard'`(CLS 반환).

---

## 검증 방법 (평가 프로토콜)

- **평가 지표/파이프라인**: 기존 `eval_linear_probe` + `metrics.py`의 BACC/AUROC/AUPR/W-F1 그대로 재사용. 출력은 현재 `PanDerm/output_dir/<dataset>_panderm_large_lp/` 구조 준용.
- **few-shot 프로토콜**: 표본이 작으므로 단일 split 대신 **반복 stratified split(예: 5~10회) 평균±표준편차** 또는 k-fold로 분산 보고. class 불균형 반영해 BACC/AUPR 중심.
- **비교 설계**: (a) 현재 baseline(마지막 CLS + 기본 probe) 대비, (b) 각 방법을 동일 split·동일 지표로 표 비교. 특징 추출은 `.npy` 캐시(`oral_diseases_features_cache/`)로 재사용해 비용 절감.
- **성공 기준**: 인접 도메인(구강 7-class 등)에서 baseline 대비 BACC/AUPR 향상, 특히 소수 클래스(OLP 등) recall 개선.
