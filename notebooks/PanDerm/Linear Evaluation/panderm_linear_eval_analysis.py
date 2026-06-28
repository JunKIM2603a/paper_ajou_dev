"""
panderm_linear_eval_analysis.py
노트북 코드를 직접 실행하여 시각화 결과물을 생성하는 스크립트.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# ──────────────────────────────────────────
# 경로 설정 (스크립트는 paper_ajou_dev 루트에서 실행)
# ──────────────────────────────────────────
DATA_ROOT   = "PanDerm/data"
OUTPUT_ROOT = "PanDerm/output_dir"
FIGURES_DIR = "PanDerm/Linear Evaluation/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

FOCUS_DATASETS = {
    "aptos2019": {
        "meta_csv": f"{DATA_ROOT}/aptos2019/Linear Evaluation/aptos2019_multiclass.csv",
        "pred_csv": f"{OUTPUT_ROOT}/aptos2019_panderm_large_lp/aptos2019_multiclass.csv",
        "label_col": "label",
        "n_classes": 5,
        "class_names": ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative"],
    },
    "Oral_Diseases": {
        "meta_csv": f"{DATA_ROOT}/Oral_Diseases/Linear Evaluation/oral_diseases_multiclass.csv",
        "pred_csv": f"{OUTPUT_ROOT}/oral_diseases_panderm_large_lp/oral_diseases_multiclass.csv",
        "label_col": "label",
        "n_classes": 7,
        "class_names": ["0: CaS", "1: CoS", "2: Gum", "3: MC", "4: OC", "5: OLP", "6: OT"],
    },
}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ──────────────────────────────────────────
# STEP 1-A: 레이블 불균형 bar chart
# ──────────────────────────────────────────
print("\n[1/5] 레이블 불균형 시각화...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("레이블 불균형 분석: aptos2019 & Oral_Diseases", fontsize=14, fontweight="bold", y=1.01)
SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
SPLITS = ["train", "val", "test"]

for row_idx, (ds_name, ds_cfg) in enumerate(FOCUS_DATASETS.items()):
    meta_df = pd.read_csv(ds_cfg["meta_csv"])
    label_col = ds_cfg["label_col"]
    class_names = ds_cfg["class_names"]
    n_classes = ds_cfg["n_classes"]
    dist = pd.crosstab(meta_df[label_col], meta_df["split"])

    for col_idx, split in enumerate(SPLITS):
        ax = axes[row_idx][col_idx]
        if split not in dist.columns:
            ax.set_visible(False)
            continue
        counts = dist[split].values
        bars = ax.bar(range(n_classes), counts, color=SPLIT_COLORS[split], alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(cnt)), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_title(f"{ds_name} – {split.upper()}", fontsize=10, fontweight="bold", color=SPLIT_COLORS[split])
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels([c.split(":")[0] for c in class_names], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("샘플 수", fontsize=8)
        ax.set_ylim(0, counts.max() * 1.2)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
out_path = f"{FIGURES_DIR}/label_imbalance_focus.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"  ✅ 저장: {out_path}")

# ──────────────────────────────────────────
# STEP 1-B: 불균형 비율 정량 계산
# ──────────────────────────────────────────
print("\n[2/5] 불균형 비율 (IR) 계산...")
print("=" * 60)
for ds_name, ds_cfg in FOCUS_DATASETS.items():
    meta_df = pd.read_csv(ds_cfg["meta_csv"])
    label_col = ds_cfg["label_col"]
    print(f"\n📌 {ds_name}")
    print(f"  {'Split':<8} {'클래스':>6} {'최다':>8} {'최소':>8} {'IR(최다/최소)':>14}")
    print("  " + "-" * 46)
    for split in ["train", "val", "test"]:
        sub = meta_df[meta_df["split"] == split][label_col].value_counts()
        if sub.empty:
            continue
        majority_cls = sub.idxmax()
        minority_cls = sub.idxmin()
        ir = sub.max() / sub.min()
        print(f"  {split:<8} {len(sub):>6}개  "
              f"{majority_cls}({sub.max():>4}개) / {minority_cls}({sub.min():>4}개)  IR={ir:.2f}x")
print("=" * 60)

# ──────────────────────────────────────────
# STEP 2: 클래스별 성능 분석
# ──────────────────────────────────────────
print("\n[3/5] 클래스별 성능 분석...")

pred_data = {}
for ds_name, ds_cfg in FOCUS_DATASETS.items():
    pred_df = pd.read_csv(ds_cfg["pred_csv"])
    y_true = pred_df["true_label"].values
    y_pred = pred_df["predicted_label"].values
    prob_cols = [c for c in pred_df.columns if c.startswith("probability_class_")]
    probs = pred_df[prob_cols].values
    pred_data[ds_name] = {"y_true": y_true, "y_pred": y_pred, "probs": probs,
                          "class_names": ds_cfg["class_names"]}

    print(f"\n{'=' * 55}")
    print(f"  {ds_name}")
    print(f"{'=' * 55}")
    print(classification_report(y_true, y_pred, target_names=ds_cfg["class_names"], zero_division=0))
    auroc = roc_auc_score(y_true, probs, multi_class="ovo", average="macro")
    aupr_macro = average_precision_score(y_true, probs, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    print(f"  Macro AUROC : {auroc:.4f}")
    print(f"  Macro AUPR  : {aupr_macro:.4f}")
    print(f"  Balanced Acc: {bacc:.4f}")
    print(f"  Accuracy    : {acc:.4f}  (차이 = {acc - bacc:+.4f})")

# ── 클래스별 성능 히트맵
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("클래스별 성능 히트맵 (Precision / Recall / F1)", fontsize=13, fontweight="bold")

for ax, (ds_name, data) in zip(axes, pred_data.items()):
    y_true = data["y_true"]; y_pred = data["y_pred"]
    class_names = data["class_names"]; n_cls = len(class_names)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_cls)), zero_division=0)
    perf_df = pd.DataFrame({"Precision": p, "Recall": r, "F1": f},
                           index=[c.split(":")[1].strip() if ":" in c else c for c in class_names])
    sns.heatmap(perf_df, ax=ax, vmin=0.0, vmax=1.0, cmap="RdYlGn",
                annot=True, fmt=".2f", linewidths=0.5, linecolor="#ccc", cbar_kws={"shrink": 0.8})
    ax.set_title(ds_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("지표", fontsize=9); ax.set_ylabel("클래스", fontsize=9)
    ax.tick_params(axis="x", labelsize=9); ax.tick_params(axis="y", labelsize=9, rotation=0)

plt.tight_layout()
out_path = f"{FIGURES_DIR}/classwise_performance_heatmap_focus.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150); plt.close()
print(f"\n  ✅ 저장: {out_path}")

# ── Confusion Matrix 정규화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Confusion Matrix (정규화: 실제 클래스 기준)", fontsize=13, fontweight="bold")

for ax, (ds_name, data) in zip(axes, pred_data.items()):
    y_true = data["y_true"]; y_pred = data["y_pred"]
    class_names = data["class_names"]; n_cls = len(class_names)
    tick_labels = [c.split(":")[1].strip() if ":" in c else c for c in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_norm[i, j]; raw = cm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({raw})", ha="center", va="center", fontsize=7.5, color=text_color)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("예측 클래스", fontsize=9); ax.set_ylabel("실제 클래스", fontsize=9)
    ax.set_title(ds_name, fontsize=11, fontweight="bold")

plt.tight_layout()
out_path = f"{FIGURES_DIR}/confusion_matrix_normalized_focus.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150); plt.close()
print(f"  ✅ 저장: {out_path}")

# ── 오분류 패턴 텍스트 요약
print("\n  주요 오분류 패턴 (True → Predicted, 상위 오류)")
print("=" * 60)
for ds_name, data in pred_data.items():
    y_true = data["y_true"]; y_pred = data["y_pred"]
    class_names = data["class_names"]; n_cls = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    print(f"\n📌 {ds_name}")
    errors = []
    for i in range(n_cls):
        total_i = cm[i].sum()
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                errors.append({"실제": class_names[i], "예측": class_names[j],
                               "오진 수": int(cm[i, j]), "오진율": cm[i, j] / total_i})
    err_df = pd.DataFrame(errors).sort_values("오진 수", ascending=False).reset_index(drop=True)
    print(err_df.head(8).to_string(index=False))

# ──────────────────────────────────────────
# STEP 3: Class-specific AUPR + Weighted AUPR
# ──────────────────────────────────────────
print("\n[4/5] Class-specific AUPR & Weighted AUPR...")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Class-specific AUPR 비교", fontsize=13, fontweight="bold")

for ax, (ds_name, data) in zip(axes, pred_data.items()):
    y_true = data["y_true"]; probs = data["probs"]
    class_names = data["class_names"]; n_cls = len(class_names)
    classes = list(range(n_cls))
    y_bin = label_binarize(y_true, classes=classes)
    support = np.bincount(y_true, minlength=n_cls)

    per_class_aupr = []
    for i in range(n_cls):
        try:
            per_class_aupr.append(average_precision_score(y_bin[:, i], probs[:, i]))
        except Exception:
            per_class_aupr.append(float("nan"))

    macro_aupr    = np.nanmean(per_class_aupr)
    weighted_aupr = np.nansum([a * s for a, s in zip(per_class_aupr, support) if not np.isnan(a)]) / support.sum()

    print(f"\n📌 {ds_name}")
    print(f"  {'클래스':<22} {'AUPR':>8}  {'Support':>9}")
    print("  " + "-" * 42)
    for cname, aupr_i, sup in zip(class_names, per_class_aupr, support):
        flag = " ⚠️" if aupr_i < 0.6 else ""
        print(f"  {cname:<22} {aupr_i:>8.4f}  {sup:>9}{flag}")
    print("  " + "-" * 42)
    print(f"  {'Macro AUPR':<22} {macro_aupr:>8.4f}")
    print(f"  {'Weighted AUPR':<22} {weighted_aupr:>8.4f}")

    tick_labels = [c.split(":")[1].strip() if ":" in c else c for c in class_names]
    colors = ["#C44E52" if v < 0.6 else "#4C72B0" for v in per_class_aupr]
    bars = ax.bar(range(n_cls), per_class_aupr, color=colors, alpha=0.85, edgecolor="white")
    for bar, val, sup in zip(bars, per_class_aupr, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}\n(n={sup})", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(macro_aupr,    color="#DD8800", ls="--", lw=1.5, label=f"Macro AUPR={macro_aupr:.3f}")
    ax.axhline(weighted_aupr, color="#228B22", ls=":",  lw=1.5, label=f"Weighted AUPR={weighted_aupr:.3f}")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUPR", fontsize=9)
    ax.set_title(ds_name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

print("=" * 65)
plt.tight_layout()
out_path = f"{FIGURES_DIR}/class_specific_aupr_focus.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150); plt.close()
print(f"  ✅ 저장: {out_path}")

# ── Balanced Accuracy vs Accuracy 갭
fig, ax = plt.subplots(figsize=(8, 5))
ds_names = list(pred_data.keys())
accs  = [accuracy_score(d["y_true"], d["y_pred"]) for d in pred_data.values()]
baccs = [balanced_accuracy_score(d["y_true"], d["y_pred"]) for d in pred_data.values()]
gaps  = [a - b for a, b in zip(accs, baccs)]
x = np.arange(len(ds_names)); width = 0.32
bars1 = ax.bar(x - width / 2, accs,  width, label="Accuracy",         color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width / 2, baccs, width, label="Balanced Accuracy", color="#C44E52", alpha=0.85)
for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#4C72B0", fontweight="bold")
for bar, val in zip(bars2, baccs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#C44E52", fontweight="bold")
for i, (a, b, g) in enumerate(zip(accs, baccs, gaps)):
    ax.annotate("", xy=(i + width / 2, b), xytext=(i - width / 2, a),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(i + width / 2 + 0.03, (a + b) / 2,
            f"갭\n{g:+.3f}", ha="left", va="center", fontsize=8, color="gray")
ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=10)
ax.set_ylabel("지표 값", fontsize=10)
ax.set_title("Accuracy vs Balanced Accuracy 갭\n(갭이 클수록 클래스 불균형 편향 심각)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.axhline(0.8, color="#999", ls="--", lw=0.8, alpha=0.5)
plt.tight_layout()
out_path = f"{FIGURES_DIR}/acc_vs_bacc_gap_focus.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150); plt.close()
print(f"  ✅ 저장: {out_path}")

# ──────────────────────────────────────────
# STEP 4: Class Weights 계산
# ──────────────────────────────────────────
print("\n[5/5] Class Weights 계산 (Train 기준)...")
print("=" * 60)
for ds_name, ds_cfg in FOCUS_DATASETS.items():
    meta_df = pd.read_csv(ds_cfg["meta_csv"])
    train_df = meta_df[meta_df["split"] == "train"]
    label_col = ds_cfg["label_col"]; class_names = ds_cfg["class_names"]
    n_cls = ds_cfg["n_classes"]; classes = np.arange(n_cls)
    y_train = train_df[label_col].values
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    train_counts = np.bincount(y_train, minlength=n_cls)
    print(f"\n📌 {ds_name} (Train 샘플 수: {len(y_train)})")
    print(f"  {'클래스':<22} {'Train 수':>10} {'Weight':>10}")
    print("  " + "-" * 44)
    for cname, cnt, w in zip(class_names, train_counts, weights):
        print(f"  {cname:<22} {cnt:>10}   {w:>10.4f}")
    weights_str = ", ".join([f"{w:.4f}" for w in weights])
    print(f"\n  # PyTorch 사용 예시:")
    print(f"  class_weights = torch.tensor([{weights_str}], dtype=torch.float)")
    print(f"  criterion = nn.CrossEntropyLoss(weight=class_weights)")
print("=" * 60)

print("\n\n✅ 모든 분석 완료!")
print(f"생성된 파일 목록: {FIGURES_DIR}/")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  - {f}")
