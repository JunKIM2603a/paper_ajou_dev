#!/usr/bin/env python
"""BitFit · LN-tuning 학습 러너 — §2-3 6방법 비교표의 빈 행을 채운다.

LoRA/Full-FT 와 **동일 프로토콜**(같은 split·eval transform·지표 정의)로
frozen PanDerm 위에서 두 PEFT 방법을 학습한다. LoRA notebook 경로(peft_ft_utils)를 재사용하며
결과는 LoRA 와 동일한 저장 규약으로 떨어져 collect_peft_table.py / 비교 노트북이 그대로 읽는다.

사용 예 (PanDerm conda env 에서):
    conda activate PanDerm
    # 개별
    python run_peft_selective.py --dataset oral_diseases --method bitfit
    python run_peft_selective.py --dataset aptos2019     --method ln_tuning
    # 2×2 전체 (aptos·oral × bitfit·ln_tuning)
    python run_peft_selective.py --dataset all --method all
    # 1 epoch 스모크(파이프라인/메모리 확인)
    python run_peft_selective.py --dataset oral_diseases --method bitfit --smoke
    # 6GB GPU(개발 1060) OOM 시 배치 축소
    python run_peft_selective.py --dataset aptos2019 --method all --batch_size 4 --accum_steps 16

GPU 선택: CUDA_VISIBLE_DEVICES=0 python run_peft_selective.py ...
집(3070)/개발(1060 6GB) 어디서 돌려도 되며, OOM 나면 --batch_size 4 --accum_steps 16 로 유효배치 64 유지.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 이 스크립트 위치를 sys.path 에 넣어 peft_ft_utils 를 import (노트북과 동일 모듈)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import peft_ft_utils as U  # noqa: E402

ALL_DATASETS = ["aptos2019", "oral_diseases"]
ALL_METHODS = ["bitfit", "ln_tuning"]


def parse_args():
    p = argparse.ArgumentParser(description="BitFit/LN-tuning PEFT 러너 (§2-3 확장)")
    p.add_argument("--dataset", default="oral_diseases",
                   choices=ALL_DATASETS + ["all"], help="대상 데이터셋 (all=둘 다)")
    p.add_argument("--method", default="all",
                   choices=ALL_METHODS + ["all"], help="PEFT 방법 (all=둘 다)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=4)  # 유효 배치 = batch*accum = 32
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--adapt_lr", type=float, default=1e-4)  # bias/LN lr
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_grad_checkpoint", action="store_true",
                   help="gradient checkpointing 끄기(메모리 여유 있을 때 약간 빠름)")
    p.add_argument("--smoke", action="store_true", help="1 epoch 파이프라인 확인")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    methods = ALL_METHODS if args.method == "all" else [args.method]
    epochs = 1 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  datasets={datasets}  methods={methods}  "
          f"epochs={epochs} eff_batch={args.batch_size * args.accum_steps}")
    if dev == "cpu":
        print("[경고] CUDA 미탐지 — CPU 로는 매우 느림. GPU 머신에서 실행 권장.")

    summary = []
    for ds in datasets:
        for m in methods:
            print("\n" + "=" * 70)
            print(f"▶ {ds} · {m}")
            print("=" * 70)
            res = U.run_peft_experiment(
                ds, m,
                batch_size=args.batch_size, accum_steps=args.accum_steps,
                epochs=epochs, patience=patience,
                head_lr=args.head_lr, adapt_lr=args.adapt_lr,
                use_grad_checkpoint=not args.no_grad_checkpoint,
                seed=args.seed, save=True, verbose=True,
            )
            mt = res["metrics"]
            row = {
                "dataset": ds, "method": m,
                "bacc": round(mt["bacc"], 4), "aupr": round(mt["aupr"], 4),
                "acc": round(mt["acc"], 4), "trainable_pct": round(res["trainable_pct"], 4),
            }
            # 소수 클래스 recall (oral=OLP, aptos=severe/proliferative)
            cfg = U.DATASETS[ds]
            row["minority_recall"] = {
                name: round(mt["per_class_recall"][str(idx)], 4)
                for name, idx in cfg["minority_classes"].items()
            }
            summary.append(row)
            print(f"  → BACC={row['bacc']} AUPR={row['aupr']} ACC={row['acc']} "
                  f"trainable={row['trainable_pct']}%  minority={row['minority_recall']}")

    print("\n" + "=" * 70)
    print("요약 (§2-3 표에 반영할 값):")
    for r in summary:
        print(f"  {r['dataset']:14s} {r['method']:10s}  "
              f"BACC={r['bacc']}  AUPR={r['aupr']}  ACC={r['acc']}  "
              f"trainable%={r['trainable_pct']}  minority={r['minority_recall']}")
    print("\n다음: python collect_peft_table.py  → 6방법 비교표 CSV/markdown 생성")


if __name__ == "__main__":
    main()
