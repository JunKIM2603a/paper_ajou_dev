#!/usr/bin/env python
"""§2-3 6방법 비교표 수집 — baseline / LoRA(pure,fusion) / BitFit / LN-tuning / Full-FT.

각 방법의 test 예측 CSV 를 **동일 지표 정의**(peft_ft_utils.metrics_from_prediction_csv)로
재계산해 apples-to-apples 표를 만든다. 아직 학습 안 된 방법은 자동으로 TBD.

산출:
  - PanDerm/output_dir/peft_ft_comparison_<ds>.csv  (전체 지표+CI)
  - stdout 에 회의록 §2-3 형식 markdown 표(aptos: trainable%, oral: OLP recall)

사용:
    conda activate PanDerm
    python collect_peft_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import peft_ft_utils as U  # noqa: E402

OUTPUT_ROOT = U.OUTPUT_ROOT

# (표시명, 결과 dir suffix, 예측 CSV 파일명 규칙). Full-FT 는 test.csv 규칙이 달라 별도 처리.
METHOD_SPECS = [
    ("LoRA (pure)",   "lora_lp",           "{ds}_pure_test_predictions.csv"),
    ("LoRA (fusion)", "lora_multilayer_lp", "{ds}_fusion_test_predictions.csv"),
    ("BitFit",        "bitfit_lp",          "{ds}_bitfit_test_predictions.csv"),
    ("LN-tuning",     "ln_tuning_lp",       "{ds}_ln_tuning_test_predictions.csv"),
]


def _safe(path, K):
    p = Path(path)
    if not p.exists():
        return None
    try:
        m, ci = U.metrics_from_prediction_csv(p, K)
        return {"metrics": m, "ci": ci}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {path}: {e}")
        return None


def _trainable_pct(result_dir):
    meta = Path(result_dir) / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("trainable_pct")
        except Exception:  # noqa: BLE001
            return None
    return None


def collect(dsk):
    cfg = U.DATASETS[dsk]
    K = cfg["num_classes"]
    rows = []

    base = U.load_baseline_row(cfg)
    rows.append({"method": "Linear Eval (baseline)", "bacc": base["bacc"], "aupr": base["aupr"],
                 "acc": base["acc"], "auroc": base.get("auroc", np.nan),
                 "weighted_f1": base.get("weighted_f1", np.nan),
                 "trainable_pct": 0.0, "minority": {}, "src": "anchor"})

    for name, suffix, fname in METHOD_SPECS:
        rdir = OUTPUT_ROOT / f"{dsk}_{suffix}"
        r = _safe(rdir / fname.format(ds=dsk), K)
        if r:
            m = r["metrics"]
            rows.append({"method": name, "bacc": m["bacc"], "aupr": m["aupr"], "acc": m["acc"],
                         "auroc": m["auroc"], "weighted_f1": m["weighted_f1"],
                         "trainable_pct": _trainable_pct(rdir),
                         "minority": {n: m["per_class_recall"][str(i)]
                                      for n, i in cfg["minority_classes"].items()},
                         "src": "pred_csv"})
        else:
            rows.append({"method": name, "bacc": np.nan, "aupr": np.nan, "acc": np.nan,
                         "auroc": np.nan, "weighted_f1": np.nan, "trainable_pct": None,
                         "minority": {}, "src": "TBD"})

    # Full-FT: run_class_finetuning 의 test.csv
    ft_dir = OUTPUT_ROOT / f"{dsk}_full_ft"
    r = _safe(ft_dir / "test.csv", K)
    if r:
        m = r["metrics"]
        rows.append({"method": "Full FT", "bacc": m["bacc"], "aupr": m["aupr"], "acc": m["acc"],
                     "auroc": m["auroc"], "weighted_f1": m["weighted_f1"], "trainable_pct": 100.0,
                     "minority": {n: m["per_class_recall"][str(i)]
                                  for n, i in cfg["minority_classes"].items()},
                     "src": "test.csv"})
    else:
        rows.append({"method": "Full FT", "bacc": np.nan, "aupr": np.nan, "acc": np.nan,
                     "auroc": np.nan, "weighted_f1": np.nan, "trainable_pct": 100.0,
                     "minority": {}, "src": "TBD"})

    return pd.DataFrame(rows)


def _fmt(x, nd=3):
    return "TBD" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def print_minutes_table(dsk, df):
    """회의록 §2-3 형식 markdown. aptos=trainable%, oral=소수클래스 recall."""
    print(f"\n### {dsk}\n")
    if dsk == "aptos2019":
        print("| 방법 | Balanced Acc | Macro AUPR | Acc | trainable % |")
        print("| --- | --- | --- | --- | --- |")
        for _, r in df.iterrows():
            tp = r["trainable_pct"]
            tp_s = "~0" if tp == 0.0 else ("100%" if tp == 100.0 else _fmt(tp, 2) + "%" if tp is not None else "TBD")
            print(f"| {r['method']} | {_fmt(r['bacc'])} | {_fmt(r['aupr'])} | {_fmt(r['acc'])} | {tp_s} |")
    else:
        print("| 방법 | Balanced Acc | Macro AUPR | Acc | OLP Recall |")
        print("| --- | --- | --- | --- | --- |")
        for _, r in df.iterrows():
            olp = r["minority"].get("OLP") if isinstance(r["minority"], dict) else None
            print(f"| {r['method']} | {_fmt(r['bacc'])} | {_fmt(r['aupr'])} | {_fmt(r['acc'])} | {_fmt(olp, 2)} |")


def main():
    for dsk in ["aptos2019", "oral_diseases"]:
        df = collect(dsk)
        out = OUTPUT_ROOT / f"peft_ft_comparison_{dsk}.csv"
        save_df = df.copy()
        save_df["minority"] = save_df["minority"].apply(lambda d: json.dumps(d, ensure_ascii=False))
        save_df.round(4).to_csv(out, index=False)
        print(f"saved {out}")
        print_minutes_table(dsk, df)
    print("\n완료. 위 markdown 표를 docs/.../2026-07-17_10th_meeting.md §2-3 에 반영하세요.")


if __name__ == "__main__":
    main()
