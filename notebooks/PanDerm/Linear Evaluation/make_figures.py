#!/usr/bin/env python
"""W10 논문 Figure 초안 3종 생성 (§③-a/b/c).

  (a) OLP 레이어별 선형 분리 곡선   ← oral_diseases_olp_separability/pairwise_separability_linear.csv
  (b) PEFT 6방법 비교 (2 데이터셋)  ← peft_ft_comparison_{aptos2019,oral_diseases}.csv
  (c) 도메인 거리 전이 패턴         ← figures_w10/per_layer_sweep_{slug}.csv  (extract_layer_sweep.py 산출)

색: Okabe–Ito CVD-safe 팔레트. 논문용 정적 그림이라 흰 배경 단일 테마, PNG(200dpi)+PDF(벡터) 저장.
대비 WARN 색(orange·purple)은 라인+직접라벨(secondary encoding)로 relief.

사용:
    conda activate PanDerm   # 또는 matplotlib/pandas 있는 아무 env
    python make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import peft_ft_utils as U  # noqa: E402

OUT = U.OUTPUT_ROOT / "figures_w10"
OUT.mkdir(parents=True, exist_ok=True)
SEP_CSV = U.OUTPUT_ROOT / "oral_diseases_olp_separability" / "pairwise_separability_linear.csv"

# Okabe–Ito
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", orange="#E69F00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#111111", gray="#7F7F7F")
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#E3E3E3"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ────────────────────────────────────────────────────────────────────────────
# (a) OLP 레이어별 선형 분리 곡선
# ────────────────────────────────────────────────────────────────────────────
def fig_a():
    df = pd.read_csv(SEP_CSV)
    pairs = ["OLP_vs_OT", "OLP_vs_OC", "OLP_vs_MC", "OLP_vs_Gum", "OLP_vs_rest"]
    labels = {"OLP_vs_OT": "vs OT", "OLP_vs_OC": "vs OC", "OLP_vs_MC": "vs MC",
              "OLP_vs_Gum": "vs Gum", "OLP_vs_rest": "vs rest"}
    colors = {"OLP_vs_OT": OI["blue"], "OLP_vs_OC": OI["vermillion"], "OLP_vs_MC": OI["green"],
              "OLP_vs_Gum": OI["orange"], "OLP_vs_rest": OI["purple"]}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(0.0, 0.5, color="#f2f2f2", zorder=0)  # 우연(chance) 이하 영역
    ax.axhline(0.5, color=OI["gray"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.81, color="#444", lw=1.1, ls=(0, (1, 2)), zorder=1)
    ax.text(0.2, 0.505, "chance (0.5)", color=MUTED, fontsize=8.5, va="bottom")
    ax.text(0.3, 0.972, "dotted = 0.81; every OLP-vs-class pair clears it at deep layers",
            color="#333", fontsize=8.5, va="top", ha="left")

    for p in pairs:
        d = df[df["pair"] == p].sort_values("layer")
        x, y, s = d["layer"].values, d["lin_bacc_mean"].values, d["lin_bacc_std"].values
        ax.fill_between(x, y - s, y + s, color=colors[p], alpha=0.09, lw=0, zorder=2)
        ax.plot(x, y, color=colors[p], lw=2.0, marker="o", ms=3.5, zorder=3, label=labels[p])

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Transformer block (CLS token, layer index)")
    ax.set_ylabel("Linear probe balanced accuracy\n(OLP vs. class, train+val CV)")
    ax.set_title("(a) OLP is linearly separable at every deep layer")
    ax.margins(x=0)
    leg = ax.legend(loc="lower right", fontsize=9.5, ncol=1, title="OLP one-vs-",
                    title_fontsize=9, handlelength=1.6, borderaxespad=0.6)
    leg.get_title().set_color(MUTED)
    save(fig, "fig_a_olp_layerwise_separability")


# ────────────────────────────────────────────────────────────────────────────
# (b) PEFT 6방법 비교 (2 데이터셋, grouped bar: BACC + AUPR)
# ────────────────────────────────────────────────────────────────────────────
METHOD_ORDER = ["Linear Eval (baseline)", "LoRA (pure)", "LoRA (fusion)",
                "BitFit", "LN-tuning", "Full FT"]
SHORT = {"Linear Eval (baseline)": "Linear\n(base)", "LoRA (pure)": "LoRA\n(pure)",
         "LoRA (fusion)": "LoRA\n(fusion)", "BitFit": "BitFit", "LN-tuning": "LN-tune",
         "Full FT": "Full FT"}


def _panel_bars(ax, df, title, sublabels, sub_caption):
    df = df.set_index("method").reindex(METHOD_ORDER).reset_index()
    x = np.arange(len(df))
    w = 0.38
    b1 = ax.bar(x - w / 2, df["bacc"], w, color=OI["blue"], label="Balanced Acc", zorder=3)
    b2 = ax.bar(x + w / 2, df["aupr"], w, color=OI["orange"], label="Macro AUPR", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=7.2, color=INK)
    base_bacc = float(df.loc[df["method"] == "Linear Eval (baseline)", "bacc"].iloc[0])
    ax.axhline(base_bacc, color=OI["gray"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[m]}\n{sublabels[m]}" for m in df["method"]], fontsize=8.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xlabel(sub_caption, fontsize=8.5, color=MUTED, labelpad=6)


def fig_b():
    import json
    ap = pd.read_csv(U.OUTPUT_ROOT / "peft_ft_comparison_aptos2019.csv")
    orl = pd.read_csv(U.OUTPUT_ROOT / "peft_ft_comparison_oral_diseases.csv")

    # LoRA trainable% 는 meta.json 부재로 CSV 가 NaN → 노트북 실측값으로 보정
    LORA_TP = {"LoRA (pure)": 0.39, "LoRA (fusion)": 0.40}

    def tp_str(method, tp):
        if pd.isna(tp):
            tp = LORA_TP.get(method)
        if tp is None:
            return "?"
        return "~0" if tp == 0 else ("100%" if tp == 100 else f"{tp:.2f}%")

    ap_sub = {r["method"]: tp_str(r["method"], r["trainable_pct"]) for _, r in ap.iterrows()}

    def olp_str(r):
        try:
            d = json.loads(r["minority"]) if isinstance(r["minority"], str) else {}
        except Exception:
            d = {}
        if "OLP" in d:
            return f"{d['OLP']:.2f}"
        return "0.30" if r["method"].startswith("Linear") else "—"  # baseline OLP=문서값 0.30

    or_sub = {r["method"]: olp_str(r) for _, r in orl.iterrows()}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    _panel_bars(axes[0], ap, "(b) aptos2019 — far domain (test=554)",
                ap_sub, "method  ·  bottom line = trainable %")
    _panel_bars(axes[1], orl, "Oral_Diseases — near domain (test=54)",
                or_sub, "method  ·  bottom line = OLP recall (n=10)")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    fig.suptitle("PEFT spectrum: 6 methods on frozen PanDerm-Large  "
                 "(gray dashed = frozen linear-probe baseline BACC)",
                 fontsize=10.5, y=1.0, color=INK)
    save(fig, "fig_b_peft_6method_comparison")


# ────────────────────────────────────────────────────────────────────────────
# (c) 도메인 거리 전이 패턴 (per-layer val BACC, aptos vs oral)
# ────────────────────────────────────────────────────────────────────────────
def _panel_curve(ax, d, title, insight, legend_loc):
    x = d["layer"].values
    for col, name, c in [("val_bacc", "val (selection)", OI["blue"]),
                         ("test_bacc", "test (held-out)", OI["vermillion"])]:
        y = d[col].values
        ax.plot(x, y, color=c, lw=2.0, marker="o", ms=3.4, zorder=3, label=name)
        bi = int(np.argmax(y))
        ax.scatter([x[bi]], [y[bi]], s=150, facecolor="white", edgecolor=c,
                   linewidth=2.2, zorder=4, marker="*")
        ax.annotate(f"L{x[bi]}", (x[bi], y[bi]), textcoords="offset points",
                    xytext=(0, 11), ha="center", color=c, fontsize=9, fontweight="bold")
    ax.axvline(23, color=OI["gray"], lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.set_xlim(-0.5, 24.2)
    ax.set_xticks(range(0, 24, 3))
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Transformer block (layer index)")
    # 인사이트: 패널 상단 좌측 여백(초기 레이어가 낮아 비어있음)
    ax.text(0.03, 0.97, insight, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#333", fontweight="bold")
    ax.text(23.0, ax.get_ylim()[1], " L23=baseline", color=MUTED, fontsize=7.2,
            va="top", ha="left", rotation=90)
    ax.legend(loc=legend_loc, fontsize=8.8, ncol=1)


def fig_c():
    fa = OUT / "per_layer_sweep_aptos2019.csv"
    fo = OUT / "per_layer_sweep_oral_diseases.csv"
    if not (fa.exists() and fo.exists()):
        print("  [skip] per_layer_sweep CSV 없음 — extract_layer_sweep.py 먼저 실행")
        return
    ap = pd.read_csv(fa).sort_values("layer")
    orl = pd.read_csv(fo).sort_values("layer")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=False)
    axes[0].set_ylim(0.42, 0.73)
    axes[1].set_ylim(0.45, 0.88)
    _panel_curve(axes[0], ap, "(c) aptos2019 · far domain",
                 "mid-network peaks;\nlast block drops", "lower right")
    _panel_curve(axes[1], orl, "Oral_Diseases · near domain",
                 "last block already\nnear-peak on test", "lower right")
    axes[0].set_ylabel("Linear probe balanced accuracy")
    fig.suptitle("Per-layer linear-probe accuracy vs. network depth  "
                 "(★ = per-curve peak; larger domain gap → stronger mid-network advantage)",
                 fontsize=10, y=1.0, color=INK)
    save(fig, "fig_c_domain_distance_layer_transfer")


def main():
    print("Figure 초안 생성 →", OUT)
    print("(a) OLP 레이어별 분리 곡선"); fig_a()
    print("(b) PEFT 6방법 비교"); fig_b()
    print("(c) 도메인 거리 전이 패턴"); fig_c()
    print("완료.")


if __name__ == "__main__":
    main()
