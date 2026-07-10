#!/usr/bin/env bash
# Full Fine-tuning Baseline — PanDerm_Large_FT (ViT-L/16)
# 9차 미팅 `🧠 [학습] PEFT + Fine-tuning` · Full FT 파트
#
# 기존 CLI [PanDerm/classification/run_class_finetuning.py] 를 무수정 재사용한다.
# 회의록 명시 플래그: --weights (WeightedRandomSampler), --monitor recall, --layer_decay 0.65.
# RTX 3070(8GB) 대응: --batch_size 8 --update_freq 8 (유효 배치 64) + AMP.
# wandb 는 disable 플래그가 없으므로 WANDB_MODE=offline 로 우회한다.
#
# 사용법:
#   bash run_full_ft.sh aptos2019          # aptos 본 학습(50 epoch)
#   bash run_full_ft.sh oral_diseases      # oral 본 학습
#   SMOKE=1 bash run_full_ft.sh oral_diseases   # 1 epoch 스모크(파이프라인/메모리 확인)
#   BATCH=4 ACCUM=16 bash run_full_ft.sh aptos2019   # OOM 시 배치 축소
set -euo pipefail

DATASET="${1:?사용법: bash run_full_ft.sh <aptos2019|oral_diseases>}"

# ─ 데이터셋별 파라미터
case "$DATASET" in
  aptos2019)
    NB_CLASSES=5
    CSV="../data/aptos2019/Linear Evaluation/aptos2019_multiclass.csv"
    ROOT="../data/aptos2019/"
    ;;
  oral_diseases)
    NB_CLASSES=7
    CSV="../data/Oral_Diseases/Linear Evaluation/oral_diseases_multiclass.csv"
    ROOT="../data/Oral_Diseases/"
    ;;
  *)
    echo "알 수 없는 데이터셋: $DATASET (aptos2019 | oral_diseases)"; exit 1 ;;
esac

# ─ 튜너블 (환경변수로 오버라이드 가능)
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-8}"          # 유효 배치 = BATCH * ACCUM = 64
EPOCHS="${EPOCHS:-50}"
WARMUP="${WARMUP:-10}"
LR="${LR:-5e-4}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"

if [ "${SMOKE:-0}" = "1" ]; then
  EPOCHS=1; WARMUP=0
  echo "[SMOKE] 1 epoch 파이프라인/메모리 확인 모드"
fi

# 경로 설정: 이 스크립트가 어디서 실행되든 PanDerm/classification 로 이동
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_DIR="$(cd "$HERE/../../../PanDerm/classification" && pwd)"
CKPT="../checkpoint/panderm_ll_data6_checkpoint-499.pth"
OUT="../output_dir/${DATASET}_full_ft/"

cd "$CLS_DIR"
echo "cwd=$(pwd)"
echo "DATASET=$DATASET NB_CLASSES=$NB_CLASSES BATCH=$BATCH ACCUM=$ACCUM EPOCHS=$EPOCHS -> effective batch $((BATCH*ACCUM))"
echo "output=$OUT"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES="$GPU" \
conda run -n PanDerm --no-capture-output python run_class_finetuning.py \
  --model PanDerm_Large_FT \
  --pretrained_checkpoint "$CKPT" \
  --nb_classes "$NB_CLASSES" \
  --batch_size "$BATCH" --update_freq "$ACCUM" \
  --lr "$LR" --weight_decay 0.05 \
  --warmup_epochs "$WARMUP" --epochs "$EPOCHS" \
  --layer_decay 0.65 --drop_path 0.2 \
  --mixup 0.8 --cutmix 1.0 \
  --weights --monitor recall \
  --sin_pos_emb --imagenet_default_mean_and_std --no_auto_resume --TTA \
  --num_workers 8 \
  --exp_name "${DATASET} full ft" --wandb_name "${DATASET}_full_ft_s${SEED}" \
  --output_dir "$OUT" \
  --csv_path "$CSV" \
  --root_path "$ROOT" \
  --seed "$SEED"

echo "=== 완료. 결과: $OUT (checkpoint-best.pth, test.csv) ==="
