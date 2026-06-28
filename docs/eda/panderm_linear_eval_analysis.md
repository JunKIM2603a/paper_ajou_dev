# PanDerm 선형 평가 — 지표 계산 방식 및 결과 영향 분석

생성일: 2026-06-28 (최초) / 2026-06-28 (업데이트 — 지표 계산 방식 심층 분석 추가)

본 보고서는 PanDerm Linear Evaluation(`Section 4: Linear Evaluation on Image Classification Tasks`)에서
사용하는 **성능 지표의 구체적인 계산 방식**과, 그로 인해 결과에 나타나는 영향을 분석합니다.

> **분석 노트북**: `notebooks/PanDerm/Linear Evaluation/panderm_linear_eval_analysis_20260628.ipynb`  
> **지표 계산 소스**: `PanDerm/classification/panderm_model/downstream/eval_features/metrics.py`  
> **Linear Probe 파이프라인**: `PanDerm/classification/panderm_model/downstream/eval_features/linear_probe.py`

---

## 0. Linear Evaluation 파이프라인 개요

PanDerm Linear Evaluation은 다음 과정으로 진행됩니다.

```
이미지 → PanDerm ViT Encoder (freeze) → feature 추출
→ LogisticRegression 헤드 학습 (L-BFGS, CrossEntropyLoss)
→ test set 예측
→ 지표 계산 (metrics.py)
```

### 핵심 분류기 설정

| 항목 | 설명 |
|---|---|
| 기반 모델 | PyTorch `nn.Linear` (LogisticRegression 구현) |
| 손실 함수 | `CrossEntropyLoss` (표준, 클래스 가중치 없음) |
| 최적화 | `LBFGS` + `strong_wolfe` line search |
| 정규화 강도 | `C = (feat_dim × num_classes) / 100` |
| 훈련 데이터 | train + val 합산 (`combine_trainval=True`) |

> [!WARNING]
> **Loss에 클래스 가중치가 없음**: `CrossEntropyLoss(weight=None)`으로 학습되므로,
> 불균형 데이터셋(aptos2019)에서 다수 클래스로 편향된 모델이 학습될 수 있습니다.

---

## 1. 지표 계산 방식 — `get_eval_metrics` 함수

실제로 사용되는 함수는 `metrics.py`의 `get_eval_metrics()` (부트스트랩 없는 버전)입니다.

```python
# metrics.py: get_eval_metrics() (L.303-384)
bacc = balanced_accuracy_score(targets_all, preds_all)
kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
acc = accuracy_score(targets_all, preds_all)
cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

eval_metrics = {
    "acc": acc,
    "bacc": bacc,
    "kappa": kappa,
    "weighted_f1": cls_rep["weighted avg"]["f1-score"],
}

# 확률값이 있는 경우 (다중 분류)
roc_kwargs = {"multi_class": "ovo", "average": "macro"}
roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
auc_pr = average_precision_score(targets_all, probs_all, average='macro')
```

### 1-1) 분류 헤드 출력 형태 (다중 분류 vs 이진 분류)

```python
# linear_probe.py: test_linear_probe() (L.232-237)
if NUM_C == 2:                          # 이진 분류
    probs_all = predict_proba()[:, 1]   # 양성 클래스 확률만 추출 (1D)
    roc_kwargs = {}
else:                                   # 다중 분류
    probs_all = predict_proba()         # 전체 소프트맥스 확률 (2D, shape=[N, C])
    roc_kwargs = {"multi_class": "ovo", "average": "macro"}
```

### 1-2) 각 지표별 계산 방식 상세

| 지표 | 함수 | 방식 | 비고 |
|---|---|---|---|
| **Accuracy** | `accuracy_score` | 전체 정답률 | 다수 클래스에 크게 의존 |
| **Balanced Accuracy** | `balanced_accuracy_score` | 클래스별 Recall 산술평균 | 불균형 시 훨씬 낮게 나옴 |
| **Weighted F1** | `classification_report["weighted avg"]` | 클래스별 F1 × 샘플 수 가중합산 | 다수 클래스 편향 가능 |
| **AUROC** | `roc_auc_score(multi_class="ovo", average="macro")` | OvO Macro 평균 | 2진 vs 다중 계산 방식 다름 |
| **AUPR (Macro)** | `average_precision_score(average="macro")` | OvR Macro PR-AUC | 내부적으로 label_binarize |
| **Kappa** | `cohen_kappa_score(weights="quadratic")` | Quadratic weighted κ | |

---

## 2. AUROC 계산 방식: OvO vs OvR

### 다중 분류 AUROC (`multi_class="ovo"`, `average="macro"`)

- **One-vs-One**: 가능한 모든 클래스 쌍(C choose 2)에 대해 이진 AUROC 계산
- 각 쌍의 AUROC를 산술평균하여 최종 Macro AUROC 산출
- **불균형에 더 강건**: OvR과 달리 각 쌍에서 소수/다수 클래스 혼합이 일어나므로, 극단적 불균형의 영향이 완화됨

```python
# OvO Macro AUROC 계산 예시
# C = 5 클래스 → C(5,2) = 10쌍 각각 AUROC 계산 → 10개 평균
roc_auc_score(y_true, probs, multi_class="ovo", average="macro")
```

> [!NOTE]
> OvR(`multi_class="ovr"`)로 계산하면 다수 클래스 혜택이 더 크게 반영됩니다.
> PanDerm은 OvO를 사용하므로, 소수 클래스 성능 저하가 AUROC에 더 잘 반영됩니다.

---

## 3. AUPR 계산 방식: OvR Macro (`average_precision_score`)

### 다중 분류 AUPR

```python
auc_pr = average_precision_score(targets_all, probs_all, average='macro')
```

- 내부적으로 **One-vs-Rest (OvR) binarization** 수행
- 각 클래스별 OvR AUPR 계산 후 **Macro 평균**
- 소수 클래스는 낮은 AUPR → Macro 평균 하락에 직접 기여

#### 검증 (binarize 결과 동일)

```python
# scikit-learn 내부 동작 검증
direct_score == binarized_score  # → True (수치 일치)
```

---

## 4. 데이터셋별 레이블 불균형과 지표 영향

### 4-1) aptos2019 (5 클래스: No DR / Mild / Moderate / Severe / Proliferative)

| 분할 | 클래스 분포 | IR |
|---|---|---|
| Train | 0(1263) / 1(259) / 2(699) / 3(135) / 4(206) | **9.36x** |
| Val | 0(270) / 3(28) | 9.64x |
| Test | 0(272) / 3(30) | 9.07x |

**학습에 미치는 영향**:
- 클래스 가중치 없는 `CrossEntropyLoss` → 다수 클래스(No DR) 편향 학습
- 소수 클래스(Severe=3)의 Loss 기여가 압도적으로 낮음

**지표에 미치는 영향**:
```
Accuracy    : 0.814  ← 다수 클래스가 정답률을 끌어올림
Balanced Acc: 0.628  ← 소수 클래스 Recall 반영으로 급락
갭          : +0.186  ← 심각한 편향 신호
```

**클래스별 성능 (Test)**:

| 클래스 | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| 0: No DR | 0.96 | 0.99 | 0.97 | 272 |
| 1: Mild | 0.65 | 0.54 | 0.59 | 56 |
| 2: Moderate | 0.70 | 0.81 | 0.75 | 151 |
| **3: Severe** | **0.55** | **0.37** | **0.44** | **30** |
| **4: Proliferative** | **0.59** | **0.44** | **0.51** | **45** |

**Class-specific AUPR**:

| 클래스 | AUPR | 비고 |
|:---|---:|---|
| 0: No DR | 0.9985 | |
| 1: Mild | 0.6724 | |
| 2: Moderate | 0.7837 | |
| **3: Severe** | **0.3396** | ⚠️ 가장 낮음 |
| **4: Proliferative** | **0.6014** | ⚠️ |
| Macro AUPR | 0.6791 | |
| Weighted AUPR | 0.8391 | 다수 클래스 혜택 반영 |

> [!IMPORTANT]
> `aptos2019` 결과를 해석할 때 **Accuracy(0.814)는 과대 평가**된 지표입니다.
> **Balanced Accuracy(0.628)와 Macro AUPR(0.6791)**을 주 지표로 사용해야 합니다.

---

### 4-2) Oral_Diseases (7 클래스: CaS / CoS / Gum / MC / OC / OLP / OT)

| 분할 | 클래스 분포 | IR |
|---|---|---|
| Train | 5(74) / 3(72) / 0(63) / 1(59) / 2(47) / 6(50) / 4(42) | 1.76x |
| Val | 클래스별 6~10개 | |
| Test | 클래스별 **6~10개** | |

**핵심 문제**: IR은 낮으나 Test 절대 샘플 수가 너무 적음

**지표에 미치는 영향**:
```
Test 1개 샘플 오분류 = Recall ±10~17%p 변동
```

**클래스별 성능 (Test)**:

| 클래스 | Recall | F1 | Support |
|:---|---:|---:|---:|
| 0: CaS | 0.88 | 0.93 | 8 |
| 1: CoS | 1.00 | 1.00 | 8 |
| 2: Gum | 1.00 | 0.82 | 7 |
| 3: MC | 0.67 | 0.71 | 9 |
| 4: OC | 1.00 | 0.80 | 6 |
| **5: OLP** | **0.30** | **0.43** | **10** |
| 6: OT | 0.83 | 0.71 | 6 |

**주요 오분류 패턴**:
- OLP → OT: 3건 (30%) — OT로 혼동
- OLP → OC: 2건 (20%) — OC로 혼동
- OLP 총 오진율: 70% (10개 중 7개 오분류)

**Class-specific AUPR**:

| 클래스 | AUPR | 비고 |
|:---|---:|---|
| 0: CaS | 0.9659 | |
| 1: CoS | 1.0000 | |
| 2: Gum | 0.9367 | |
| 3: MC | 0.8053 | |
| **4: OC** | **0.5802** | ⚠️ |
| **5: OLP** | **0.5824** | ⚠️ |
| 6: OT | 0.7161 | |
| Macro AUPR | 0.7981 | |

---

## 5. Weighted F1 vs Macro F1 차이

`PanDerm metrics.py`에서 보고하는 F1 지표는 **Weighted F1**입니다.

```python
"weighted_f1": cls_rep["weighted avg"]["f1-score"]
```

**Weighted F1 = Σ(클래스별 F1 × 클래스 샘플 수) / 전체 샘플 수**

| 지표 | aptos2019 | 특징 |
|---|---|---|
| Weighted F1 | 0.81 | 다수 클래스(No DR) 편향 |
| Macro F1 | 0.65 | 소수 클래스 성능 더 잘 반영 |
| Balanced Accuracy | 0.628 | 클래스별 Recall 평균 |

> [!TIP]
> 불균형 다중 분류에서는 Weighted F1 외에 **Macro F1**도 함께 확인해야
> 소수 클래스 성능 저하를 파악할 수 있습니다.

---

## 6. Accuracy vs Balanced Accuracy 갭 분석

| 데이터셋 | Accuracy | Balanced Acc | 갭 | 판단 |
|---|---:|---:|---:|---|
| aptos2019 | 0.814 | 0.628 | **+0.186** | 🔴 심각한 다수 클래스 편향 |
| Oral_Diseases | 0.778 | 0.811 | -0.033 | 🟡 소량 데이터 노이즈 (편향 아님) |

---

## 7. 지표 계산 방식이 결과 해석에 미치는 영향 요약

| 지표 | aptos2019 | Oral_Diseases | 주의점 |
|---|---|---|---|
| Accuracy | 과대 평가 | 적절 | 불균형 시 신뢰 불가 |
| Balanced Accuracy | 진실을 반영 | 적절 | 주 지표로 사용 권장 |
| Weighted F1 | 편향 | 적절 | Macro F1 보완 필요 |
| Macro AUROC (OvO) | 비교적 안정 | 소표본 노이즈 | 표준 오차 표기 권장 |
| Macro AUPR | 소수 클래스 반영 | 소표본 노이즈 | Weighted AUPR 병기 권장 |
| Class-specific AUPR | Severe(3) 최저 | OC/OLP 저조 | 반드시 클래스별 확인 |

---

## 8. 개선 방향 — Class Weights (Train 기준)

### aptos2019 권장 Class Weights

```python
# sklearn compute_class_weight('balanced') 기준
class_weights = torch.tensor([0.4057, 1.9784, 0.7330, 3.7956, 2.4874], dtype=torch.float)
# 클래스 3(Severe)에 3.80배 가중치
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Oral_Diseases 권장 Class Weights

```python
class_weights = torch.tensor([0.9229, 0.9855, 1.2371, 0.8075, 1.3844, 0.7857, 1.1629], dtype=torch.float)
# 클래스 4(OC)에 1.38배, 2(Gum)에 1.24배
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 9. 생성된 시각화 파일

| 파일 | 내용 |
|---|---|
| `figures/label_imbalance_focus.png` | 레이블 불균형 bar chart (aptos2019, Oral_Diseases) |
| `figures/classwise_performance_heatmap_focus.png` | 클래스별 P/R/F1 히트맵 (RdYlGn 색상) |
| `figures/confusion_matrix_normalized_focus.png` | 정규화 Confusion Matrix (행 기준) |
| `figures/class_specific_aupr_focus.png` | Class-specific AUPR 막대 그래프 (Macro/Weighted 기준선 포함) |
| `figures/acc_vs_bacc_gap_focus.png` | Accuracy vs Balanced Accuracy 갭 시각화 |
