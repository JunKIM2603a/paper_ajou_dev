"""PEFT(LoRA) + Fine-tuning 공통 유틸 — 9차 미팅 `🧠 [학습] PEFT + Fine-tuning`.

두 LoRA 노트북(aptos2019 / Oral_Diseases)과 최종 비교 노트북이 공유한다.
목적: PEFT·Full-FT 결과를 Linear Evaluation baseline과 **동일 프로토콜**(같은 split /
같은 eval transform / 같은 지표 정의)로 비교할 수 있게 한 곳에 통일한다.

핵심 재사용:
  - LoRA 구현(LoRALayer / monkey-patch / PanDermLoRAMultiLayer)은
    `panderm_lora_multilayer_finetune_oral_diseases_20260705.ipynb` 를 그대로 옮겨온 것.
  - 백본/데이터/eval transform 은 `PanDerm/classification` 의 기존 코드를 재사용.
정합화 교정:
  - AUROC 를 라이브러리 baseline(`get_eval_metrics`)과 동일하게 multi_class="ovo" 로 계산
    (기존 노트북은 "ovr" 이었음).
"""
from __future__ import annotations

import os
import sys
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision import transforms

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# 경로 (notebooks/PanDerm/Linear Evaluation/ 기준)
# ─────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (THIS_DIR / "../../..").resolve()
CLASSIFICATION_DIR = PROJECT_ROOT / "PanDerm" / "classification"
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

DATA_ROOT = PROJECT_ROOT / "PanDerm" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "PanDerm" / "output_dir"
CHECKPOINT = PROJECT_ROOT / "PanDerm" / "checkpoint" / "panderm_ll_data6_checkpoint-499.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 레지스트리
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "aptos2019": dict(
        name="aptos2019",
        meta_csv=DATA_ROOT / "aptos2019" / "Linear Evaluation" / "aptos2019_multiclass.csv",
        image_root=str(DATA_ROOT / "aptos2019") + "/",
        class_names=["no_dr", "mild", "moderate", "severe", "proliferative_dr"],
        num_classes=5,
        baseline_csv=OUTPUT_ROOT / "aptos2019_panderm_large_lp" / "aptos2019_panderm_large_lp_result.csv",
        minority_classes={"severe": 3, "proliferative_dr": 4},  # 관심 소수 클래스
        # aptos: 원거리 도메인, 앞선 실험에서 중간층 L11 이 val-winner
        fusion_layers=[11, 15, 19, 23],
    ),
    "oral_diseases": dict(
        name="oral_diseases",
        meta_csv=DATA_ROOT / "Oral_Diseases" / "Linear Evaluation" / "oral_diseases_multiclass.csv",
        image_root=str(DATA_ROOT / "Oral_Diseases") + "/",
        class_names=["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"],
        num_classes=7,
        baseline_csv=OUTPUT_ROOT / "oral_diseases_panderm_large_lp" / "oral_diseases_panderm_large_lp_result.csv",
        minority_classes={"OLP": 5},
        fusion_layers=[11, 15, 19, 23],
    ),
}

# PEFT 실험 변형: 순수(마지막 CLS 단독) vs 융합(중간층 결합, ablation)
VARIANTS = {
    "pure": dict(suffix="lora_lp", label="LoRA + linear head (last CLS)"),
    "fusion": dict(suffix="lora_multilayer_lp", label="LoRA + multi-layer fusion"),
}


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# LoRA 구현 (기존 노트북에서 그대로 재사용)
#   주의: PanDerm Attention.forward 는 F.linear(self.qkv.weight, ...) 로 weight 를
#   직접 읽으므로 self.qkv 모듈 치환은 무시된다 → forward 자체를 monkey-patch.
# ─────────────────────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    """저랭크 어댑터: out = (x @ A^T @ B^T) * (alpha/r). B=0 초기화 → 학습 시작 시 델타=0."""

    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return (x @ self.A.t() @ self.B.t()) * self.scale


def _lora_attention_forward(self, x, rel_pos_bias=None):
    """원본 Attention.forward(modeling_finetune.py:170)에 lora_qkv/lora_proj 델타만 추가."""
    B, N, C = x.shape
    qkv_bias = None
    if self.q_bias is not None:
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv + self.lora_qkv(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)

    if self.relative_position_bias_table is not None:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1] + 1,
               self.window_size[0] * self.window_size[1] + 1, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    if rel_pos_bias is not None:
        attn = attn + rel_pos_bias

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    out = self.proj(out) + self.lora_proj(out)
    out = self.proj_drop(out)
    return out


def inject_lora(backbone, layer_indices, r=8, alpha=16):
    """지정 블록의 attn.qkv / attn.proj 에 LoRA 부착 + forward patch. 추가 파라미터 리스트 반환."""
    dim = backbone.embed_dim
    lora_params = []
    for i in layer_indices:
        attn = backbone.blocks[i].attn
        qkv_out_dim = attn.qkv.weight.shape[0]   # all_head_dim * 3
        proj_in_dim = attn.proj.weight.shape[1]  # all_head_dim
        attn.lora_qkv = LoRALayer(dim, qkv_out_dim, r=r, alpha=alpha)
        attn.lora_proj = LoRALayer(proj_in_dim, dim, r=r, alpha=alpha)
        attn.forward = types.MethodType(_lora_attention_forward, attn)
        lora_params += list(attn.lora_qkv.parameters()) + list(attn.lora_proj.parameters())
    return lora_params


class PanDermLoRAMultiLayer(nn.Module):
    """frozen PanDerm 백본 + 지정 블록 LoRA + 여러 레이어 CLS 융합(A3) + 커스텀 head.

    fusion_layers=[23] → 순수(pure) 변형(마지막 CLS 단독 + linear head).
    fusion_layers=[11,15,19,23] → 융합(fusion) 변형.
    """

    def __init__(self, backbone, fusion_layers, num_classes,
                 lora_r=8, lora_alpha=16, lora_layers=None,
                 use_grad_checkpoint=True, head_dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.fusion_layers = sorted(fusion_layers)
        self.max_layer = self.fusion_layers[-1]
        self.use_grad_checkpoint = use_grad_checkpoint
        embed_dim = backbone.embed_dim

        for p in self.backbone.parameters():
            p.requires_grad = False

        target_layers = lora_layers if lora_layers is not None else self.fusion_layers
        self.lora_params = inject_lora(self.backbone, target_layers, r=lora_r, alpha=lora_alpha)
        for p in self.lora_params:
            p.requires_grad = True

        # 블록 출력은 최종 LayerNorm 이전이라 레이어마다 스케일이 다름 → 레이어별 학습 LN 으로 보정
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in self.fusion_layers])
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embed_dim * len(self.fusion_layers), num_classes)

        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # backbone 은 절대 train() 으로 안 바뀌게 오버라이드
        return self

    def forward(self, x):
        bb = self.backbone
        x = bb.patch_embed(x)
        B = x.shape[0]
        cls_tokens = bb.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + bb.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).detach()
        x = bb.pos_drop(x)
        rel_pos_bias = bb.rel_pos_bias() if bb.rel_pos_bias is not None else None

        collected = {}
        for i in range(self.max_layer + 1):  # max_layer 이후 블록은 실행 불필요 (조기 종료)
            blk = bb.blocks[i]
            if self.use_grad_checkpoint and self.training:
                x = grad_checkpoint(blk, x, rel_pos_bias, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            if i in self.fusion_layers:
                collected[i] = x[:, 0]

        fused = torch.cat([ln(collected[i]) for ln, i in zip(self.layer_norms, self.fusion_layers)], dim=1)
        fused = self.dropout(fused)
        return self.head(fused)

    def trainable_parameter_groups(self, head_lr=1e-3, lora_lr=1e-4, weight_decay=1e-4):
        lora_and_ln = list(self.layer_norms.parameters()) + self.lora_params
        return [
            {"params": self.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
            {"params": lora_and_ln, "lr": lora_lr, "weight_decay": weight_decay},
        ]

    def trainable_param_names(self):
        return {n for n, p in self.named_parameters() if p.requires_grad}

    def trainable_state_dict(self):
        names = self.trainable_param_names()
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd):
        self.load_state_dict(sd, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# 백본 / 데이터 / 지표
# ─────────────────────────────────────────────────────────────────────────────
def load_backbone(checkpoint=CHECKPOINT):
    """frozen PanDerm_Large_LP 백본 + baseline 과 동일한 eval transform."""
    import argparse
    from models.builder import get_encoder
    args = argparse.Namespace(pretrained_checkpoint=str(checkpoint))
    backbone, eval_transform = get_encoder(args, "PanDerm_Large_LP")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, eval_transform


def build_transforms():
    """train=약한 증강, eval=baseline 과 동일(Resize256+CenterCrop224, imagenet norm)."""
    from models.builder import get_norm_constants, get_eval_transforms
    mean, std = get_norm_constants("imagenet")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = get_eval_transforms(which_img_norm="imagenet")
    return train_transform, eval_transform


def build_loaders(cfg, batch_size=8, eval_batch_size=32, num_workers=4):
    """Derm_Dataset(split 컬럼) 로 train/val/test DataLoader 생성."""
    from datasets.derm_data import Derm_Dataset
    train_transform, eval_transform = build_transforms()
    df_all = pd.read_csv(cfg["meta_csv"])
    root = cfg["image_root"]
    ds_train = Derm_Dataset(df=df_all, root=root, train=True, transforms=train_transform, binary=False)
    ds_val = Derm_Dataset(df=df_all, root=root, val=True, transforms=eval_transform, binary=False)
    ds_test = Derm_Dataset(df=df_all, root=root, test=True, transforms=eval_transform, binary=False)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return df_all, (train_loader, val_loader, test_loader)


def compute_class_weights(df_all, cfg, device):
    """축 B1: balanced class weight (train 분포 기반)."""
    counts = df_all[df_all["split"] == "train"]["label"].value_counts().sort_index().values
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32).to(device)


def compute_metrics(y_true, y_pred, probs, num_classes):
    """라이브러리 get_eval_metrics 와 동일 정의. AUROC 는 ovo/macro (baseline 정합)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs)
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    wf1 = report["weighted avg"]["f1-score"]
    # AUROC: 라이브러리(test_linear_probe)는 multi_class="ovo", average="macro"
    auroc = roc_auc_score(y_true, probs, multi_class="ovo", average="macro")
    y_true_oh = label_binarize(y_true, classes=list(range(num_classes)))
    aupr = average_precision_score(y_true_oh, probs, average="macro")
    per_class_recall = {str(c): report.get(str(c), {}).get("recall", float("nan"))
                        for c in range(num_classes)}
    return {"bacc": bacc, "acc": acc, "weighted_f1": wf1, "auroc": auroc, "aupr": aupr,
            "per_class_recall": per_class_recall}


def bootstrap_ci(y_true, y_pred, probs, num_classes, n_boot=1000, seed=0, alpha=0.05):
    """test 재표본(복원추출)으로 BACC / Macro AUPR 의 (2.5%, 97.5%) CI."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs)
    n = len(y_true)
    baccs, auprs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y_true[idx], y_pred[idx], probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        baccs.append(balanced_accuracy_score(yt, yp))
        try:
            yt_oh = label_binarize(yt, classes=list(range(num_classes)))
            auprs.append(average_precision_score(yt_oh, pr, average="macro"))
        except ValueError:
            pass
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    def ci(a):
        return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if a else (float("nan"), float("nan"))
    return {"bacc_ci": ci(baccs), "aupr_ci": ci(auprs)}


def metrics_from_prediction_csv(path, num_classes):
    """예측 CSV(true_label, predicted_label, probability_class_0..K) → 정합 지표.

    Full-FT 의 `test.csv` 와 LoRA 의 `*_test_predictions.csv` 둘 다 이 컬럼 컨벤션을 따르므로
    같은 함수로 재계산해 4자 비교를 apples-to-apples 로 맞춘다.
    """
    df = pd.read_csv(path)
    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    prob_cols = [f"probability_class_{c}" for c in range(num_classes)]
    probs = df[prob_cols].to_numpy()
    m = compute_metrics(y_true, y_pred, probs, num_classes)
    ci = bootstrap_ci(y_true, y_pred, probs, num_classes)
    return m, ci


def load_baseline_row(cfg):
    """Linear Eval baseline result CSV 한 행을 dict 로."""
    row = pd.read_csv(cfg["baseline_csv"]).iloc[0]
    return {"bacc": float(row["BACC"]), "auroc": float(row["AUROC"]),
            "aupr": float(row["AUPR"]), "acc": float(row["ACC"]),
            "weighted_f1": float(row["W_F1"])}


@torch.no_grad()
def predict(model, loader, device, use_amp=True):
    """(y_true, y_pred, probs) 반환."""
    model.eval()
    ys, preds, all_probs = [], [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)
        preds.append(probs.argmax(1).cpu())
        all_probs.append(probs.cpu())
        ys.append(labels)
    return (torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(all_probs).numpy())


def train_lora(model, loaders, class_weights, device,
               epochs=40, accum_steps=4, head_lr=1e-3, lora_lr=1e-4,
               weight_decay=1e-4, patience=10, use_amp=True, verbose=True):
    """LoRA 학습 루프. val Balanced Accuracy 기준 best(trainable state) 유지."""
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.trainable_parameter_groups(head_lr, lora_lr, weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_val_bacc, best_state, patience_counter, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * accum_steps

        yv, pv, _ = predict(model, val_loader, device, use_amp)
        val_bacc = balanced_accuracy_score(yv, pv)
        train_loss = running_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_bacc": val_bacc})
        if verbose:
            print(f"epoch {epoch+1:02d}/{epochs}  train_loss={train_loss:.4f}  val_bacc={val_bacc:.4f}")
        if val_bacc > best_val_bacc:
            best_val_bacc, best_state, patience_counter = val_bacc, model.trainable_state_dict(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[early stop] epoch {epoch+1} (best val_bacc={best_val_bacc:.4f})")
                break
    return best_state, best_val_bacc, history


def run_lora_experiment(dataset_key, variant, *, lora_r=8, lora_alpha=16,
                        lora_layers=None, batch_size=8, accum_steps=4, epochs=40,
                        patience=10, use_grad_checkpoint=True, seed=0, save=True, verbose=True):
    """한 (데이터셋 × 변형) LoRA 실험을 end-to-end 수행하고 결과를 저장/반환."""
    assert dataset_key in DATASETS, dataset_key
    assert variant in VARIANTS, variant
    cfg = DATASETS[dataset_key]
    vinfo = VARIANTS[variant]
    set_seed(seed)
    device = get_device()
    K = cfg["num_classes"]

    fusion_layers = [23] if variant == "pure" else list(cfg["fusion_layers"])
    if lora_layers is None:
        lora_layers = list(range(24))  # 전 블록 LoRA (비용 작음)

    backbone, _ = load_backbone()
    model = PanDermLoRAMultiLayer(
        backbone, fusion_layers, K, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_layers=lora_layers, use_grad_checkpoint=use_grad_checkpoint,
    ).to(device)

    df_all, loaders = build_loaders(cfg, batch_size=batch_size)
    class_weights = compute_class_weights(df_all, cfg, device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{dataset_key} · {variant}] fusion_layers={fusion_layers} "
              f"trainable={trainable:,} ({100*trainable/total:.3f}%)")

    best_state, best_val_bacc, history = train_lora(
        model, loaders, class_weights, device, epochs=epochs, accum_steps=accum_steps,
        patience=patience, verbose=verbose)

    model.load_trainable_state_dict(best_state)
    y_true, y_pred, probs = predict(model, loaders[2], device)
    metrics = compute_metrics(y_true, y_pred, probs, K)
    ci = bootstrap_ci(y_true, y_pred, probs, K, seed=seed)
    baseline = load_baseline_row(cfg)

    result = {
        "dataset": dataset_key, "variant": variant, "label": vinfo["label"],
        "fusion_layers": fusion_layers, "lora_r": lora_r, "lora_alpha": lora_alpha,
        "best_val_bacc": best_val_bacc, "trainable_params": trainable,
        "metrics": metrics, "ci": ci, "baseline": baseline,
    }

    if save:
        result_dir = OUTPUT_ROOT / f"{dataset_key}_{vinfo['suffix']}"
        os.makedirs(result_dir, exist_ok=True)
        pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
        torch.save(best_state, result_dir / "best_trainable_state_dict.pt")
        comp = pd.DataFrame([
            {"method": "baseline (frozen CLS + linear probe)", **baseline},
            {"method": vinfo["label"], "bacc": metrics["bacc"], "auroc": metrics["auroc"],
             "aupr": metrics["aupr"], "acc": metrics["acc"], "weighted_f1": metrics["weighted_f1"]},
        ]).set_index("method").round(4)
        comp.to_csv(result_dir / "comparison_vs_baseline.csv")
        # test 예측 CSV (Full-FT test.csv 와 동일 컬럼 컨벤션)
        pred_df = pd.DataFrame({"true_label": y_true, "predicted_label": y_pred})
        for c in range(K):
            pred_df[f"probability_class_{c}"] = probs[:, c]
        pred_df.to_csv(result_dir / f"{dataset_key}_{variant}_test_predictions.csv", index=False)
        result["result_dir"] = str(result_dir)
        if verbose:
            print(f"저장: {result_dir}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# self-test: baseline 재현(±0.02) — frozen 마지막 CLS + linear probe
# ─────────────────────────────────────────────────────────────────────────────
def reproduce_baseline(dataset_key, seed=0, verbose=True):
    """라이브러리 eval_linear_probe 로 baseline 을 재현해 앵커와 비교(정합성 확인)."""
    from panderm_model.downstream.extract_features import extract_features_from_dataloader
    from panderm_model.downstream.eval_features.linear_probe import eval_linear_probe
    from datasets.derm_data import Derm_Dataset
    import argparse

    cfg = DATASETS[dataset_key]
    device = get_device()
    backbone, eval_transform = load_backbone()
    backbone.to(device)
    df_all = pd.read_csv(cfg["meta_csv"])
    root = cfg["image_root"]

    def feats(split_flag):
        ds = Derm_Dataset(df=df_all, root=root, transforms=eval_transform, binary=False, **split_flag)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        out = extract_features_from_dataloader(argparse.Namespace(), backbone, dl)
        return (torch.tensor(out["embeddings"]).float(),
                torch.tensor(out["labels"]).long(),
                out["filenames"])

    tr_f, tr_y, _ = feats({"train": True})
    va_f, va_y, _ = feats({"val": True})
    te_f, te_y, te_fn = feats({"test": True})

    # get_eval_metrics 는 out_dir 에 예측 CSV/confusion PNG 를 쓰므로 임시 경로 필요
    tmp_dir = OUTPUT_ROOT / f"_baseline_reproduce_tmp" / cfg["name"]
    os.makedirs(tmp_dir, exist_ok=True)
    metrics, _ = eval_linear_probe(
        train_feats=tr_f, train_labels=tr_y, valid_feats=None, valid_labels=va_y,
        test_feats=te_f, test_labels=te_y, test_filenames=te_fn,
        max_iter=1000, verbose=False, seed=seed, out_dir=str(tmp_dir), dataset_name=cfg["name"],
    )
    base = load_baseline_row(cfg)
    got = {"bacc": metrics["lin_bacc"], "aupr": metrics["lin_aupr"], "acc": metrics["lin_acc"]}
    if verbose:
        print(f"[{dataset_key}] reproduce BACC={got['bacc']:.4f} (anchor {base['bacc']:.4f}) "
              f"AUPR={got['aupr']:.4f} (anchor {base['aupr']:.4f})")
    ok = abs(got["bacc"] - base["bacc"]) <= 0.02 and abs(got["aupr"] - base["aupr"]) <= 0.02
    return ok, got, base


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CHECKPOINT exists:", CHECKPOINT.exists())
    for k in DATASETS:
        ok, got, base = reproduce_baseline(k)
        print(f"  {k}: reproduce {'OK' if ok else 'MISMATCH'}  {got}")
