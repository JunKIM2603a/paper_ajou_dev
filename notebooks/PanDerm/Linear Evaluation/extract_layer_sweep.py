#!/usr/bin/env python
"""도메인 거리 전이 패턴(Figure c) 데이터 재생성 — aptos·oral 24-블록 CLS 레이어 스윕.

배경: `panderm_multilayer_fusion_layer_exploration_20260708.ipynb` 가 per_layer_sweep CSV 를
`{slug}_multilayer_fusion/` 에 저장했으나 해당 디렉토리가 정리돼 사라짐. Figure 초안을 위해
동일 방법론(24블록 pre-norm CLS → StandardScaler(train) → linear probe → val/test BACC)으로
스윕을 재계산한다. 모델/데이터 로드는 학습에서 검증된 `peft_ft_utils` 를 재사용.

산출: PanDerm/output_dir/figures_w10/per_layer_sweep_{slug}.csv  (layer, val_bacc, test_bacc, ...)
검증: L23(마지막 CLS) test BACC 가 문서 baseline(aptos 0.628 / oral 0.811)과 근사하면 방법론 OK.

사용:
    conda activate PanDerm
    CUDA_VISIBLE_DEVICES=0 python extract_layer_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import peft_ft_utils as U  # noqa: E402  (모델/데이터/경로 재사용)

DEPTH = 24
SEED = 0
OUT_DIR = U.OUTPUT_ROOT / "figures_w10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_BASELINE = {"aptos2019": 0.628, "oral_diseases": 0.811}  # 문서 baseline BACC (L23 검증용)


def build_eval_loaders(cfg, batch_size=32, num_workers=2):
    """eval transform · no-shuffle · no-drop 로더 (특징 추출용, 3 split 모두)."""
    from datasets.derm_data import Derm_Dataset
    _, eval_transform = U.build_transforms()
    df_all = pd.read_csv(cfg["meta_csv"])
    root = cfg["image_root"]
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = Derm_Dataset(df=df_all, root=root, transforms=eval_transform,
                          binary=False, **{split: True})
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loaders


@torch.inference_mode()
def extract_split(model, loader, device):
    """블록별 forward hook 으로 pre-norm CLS 수집 → pre[N,24,1024], labels[N]."""
    buf = {i: [] for i in range(DEPTH)}

    def make_hook(i):
        def hook(_m, _in, out):          # out: (B,197,1024) pre-self.norm
            buf[i].append(out[:, 0].detach().to("cpu").clone())  # CLS 만 복사(뷰는 전체 seq 고정→OOM)
        return hook

    handles = [model.blocks[i].register_forward_hook(make_hook(i)) for i in range(DEPTH)]
    labels = []
    try:
        for batch in loader:
            x, y = batch[0], batch[1]
            model.forward_features(x.to(device), is_train=False)  # 전 블록 통과 → 훅 발화
            labels.append(torch.as_tensor(y))
    finally:
        for h in handles:
            h.remove()
    pre = np.stack([torch.cat(buf[i]).numpy() for i in range(DEPTH)], axis=1).astype(np.float32)
    return pre, torch.cat(labels).numpy().astype(np.int64)


def sweep_dataset(model, ds_key, device):
    cfg = U.DATASETS[ds_key]
    slug = cfg["name"]
    print(f"\n=== {slug} 특징 추출 ===")
    loaders = build_eval_loaders(cfg)
    feats, ys = {}, {}
    for split, loader in loaders.items():
        t0 = time.time()
        feats[split], ys[split] = extract_split(model, loader, device)
        print(f"  [{slug}/{split}] pre{feats[split].shape} {time.time()-t0:.0f}s")

    rows = []
    for i in range(DEPTH):
        sc = StandardScaler().fit(feats["train"][:, i, :])
        Xtr = sc.transform(feats["train"][:, i, :])
        Xval = sc.transform(feats["val"][:, i, :])
        Xte = sc.transform(feats["test"][:, i, :])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(Xtr, ys["train"])
        val_bacc = balanced_accuracy_score(ys["val"], clf.predict(Xval))
        test_bacc = balanced_accuracy_score(ys["test"], clf.predict(Xte))
        rows.append(dict(layer=i, val_bacc=val_bacc, test_bacc=test_bacc))
    sweep = pd.DataFrame(rows)

    best_val = int(sweep.sort_values("val_bacc", ascending=False)["layer"].iloc[0])
    l23 = float(sweep.loc[sweep["layer"] == DEPTH - 1, "test_bacc"].iloc[0])
    doc = DOC_BASELINE[slug]
    ok = "OK" if abs(l23 - doc) <= 0.03 else "CHECK"
    print(f"  best_val_layer=L{best_val}  L23_test_bacc={l23:.4f} (doc {doc}, |Δ|={abs(l23-doc):.4f} {ok})")

    out = OUT_DIR / f"per_layer_sweep_{slug}.csv"
    sweep.round(6).to_csv(out, index=False)
    print(f"  saved {out}")
    return sweep, best_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model, _ = U.load_backbone()
    model.to(device).eval()
    for ds_key in ["aptos2019", "oral_diseases"]:
        sweep_dataset(model, ds_key, device)
    print("\n완료 — figures_w10/per_layer_sweep_*.csv 생성. 다음: 플롯 스크립트.")


if __name__ == "__main__":
    main()
