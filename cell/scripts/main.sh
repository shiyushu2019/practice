#!/usr/bin/env bash
# 全量训练 + 验证 (使用已准备好的 mixed 数据集; 不解压或重建数据)
# YOLO26n 消融五组: baseline / +Focal Loss / +CBAM / +DAMS-Net / +CBAM+Focal+DAMS
set -e

DEVICE="${DEVICE:-0}"
DATA_YAML="dataset1/mixed_blood_cell.yaml"
EPOCHS=100
DATASET_ARCHIVE="dataset1.7z"
DATASET_SHA256="1742c29fe25727ca576046e9100e40ffcedc173b9d5b69bf92e5f70b1ae6d9f5"

if [ ! -f "$DATASET_ARCHIVE" ]; then
  echo "缺少数据集归档 $DATASET_ARCHIVE，无法验证 SHA-256。" >&2
  exit 1
fi

if [ "$(sha256sum "$DATASET_ARCHIVE" | cut -d' ' -f1)" != "$DATASET_SHA256" ]; then
  echo "$DATASET_ARCHIVE SHA-256 校验失败；拒绝训练。" >&2
  exit 1
fi
echo "$DATASET_ARCHIVE SHA-256 校验通过。"

if [ ! -f "$DATA_YAML" ]; then
  echo "缺少现成数据集配置 $DATA_YAML；请先准备 mixed 数据集。此脚本不会修改或解压数据集。" >&2
  exit 1
fi

COMMON_ARGS=(
  --epochs "$EPOCHS" --batch 16 --dataset mixed --data "$DATA_YAML"
  --device "$DEVICE" --freeze-epochs 0 --optimizer Adam --lr0 0.001
  --lrf 0.01 --weight-decay 0.0005 --warmup-epochs 3
  --mosaic 0.5 --close-mosaic 15 --degrees 180 --flipud 0.5 --fliplr 0.5
  --hsv-h 0.015 --hsv-s 0.3 --hsv-v 0.2 --patience 20 --seed 0 --skip-test
)

# Train each YOLO26n ablation without overwriting legacy YOLOv8 checkpoints.
python train.py --variant baseline --out yolo26n_ab_baseline "${COMMON_ARGS[@]}"
python train.py --variant focal --out yolo26n_ab_focal --gamma 2.0 --alpha 0.25 "${COMMON_ARGS[@]}"
python train.py --variant cbam --out yolo26n_ab_cbam "${COMMON_ARGS[@]}"
python train.py --variant dams --out yolo26n_ab_dams "${COMMON_ARGS[@]}"
python train.py --variant cf --out yolo26n_ab_cf --gamma 2.0 --alpha 0.25 "${COMMON_ARGS[@]}"

# Compare all variants on the same held-out test set.
python eval.py --split test --data "$DATA_YAML" --device "$DEVICE" --no-plots \
  --weights weights/yolo26n_ab_baseline.pt,weights/yolo26n_ab_focal.pt,weights/yolo26n_ab_cbam.pt,weights/yolo26n_ab_dams.pt,weights/yolo26n_ab_cf.pt
