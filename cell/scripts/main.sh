#!/usr/bin/env bash
# 全量训练 + 验证 + 推理 (数据集: 混合 BCCD + TXL-PBC; GPU 用 DEVICE 环境变量指定, 默认 3)
# 消融五组: baseline / +Focal Loss / +CBAM / +DAMS-Net / CF-YOLOv8; 各阶段需依次执行, 不要并行。
set -e

DEVICE="${DEVICE:-3}"
DATASET="mixed"
EPOCHS=100

# 数据准备: mixed 依赖 BCCD 预处理集 (TXL/mixed 会由 train/eval/inf 内部自动构建)
if [ ! -d dataset1/BCCD_Dataset_preprocessed ]; then
  if [ ! -d dataset1/BCCD_Dataset ]; then
    echo "错误: 缺少 dataset1/BCCD_Dataset (原始 BCCD), 无法构建预处理集" >&2
    exit 1
  fi
  echo "[prep] 构建 BCCD 预处理集..."
  python preprocess.py
fi

# 阶段1: 基线对照 YOLOv8n
python train.py --variant baseline --epochs "$EPOCHS" --dataset "$DATASET" --device "$DEVICE"

# 阶段2: YOLOv8n + Focal Loss (消融)
python train.py --variant focal --epochs "$EPOCHS" --dataset "$DATASET" --device "$DEVICE"

# 阶段3: YOLOv8n + CBAM
python train.py --variant cbam --epochs "$EPOCHS" --dataset "$DATASET" --device "$DEVICE"

# 阶段4: YOLOv8n + DAMS-Net (消融)
python train.py --variant dams --epochs "$EPOCHS" --dataset "$DATASET" --device "$DEVICE"

# 阶段5: 完整 CF-YOLOv8 (CBAM + Focal + DAMS-Net), 按文档配置 (Adam/lr0=0.001/batch16/patience10/先冻结后解冻)
python train.py --variant cf --epochs "$EPOCHS" --dataset "$DATASET" --device "$DEVICE" \
  --gamma 2.0 --alpha 0.25 --batch 16 --freeze-epochs 10

# 阶段6: 在混合测试集上验证所有变体
python eval.py --split test --dataset "$DATASET" --device "$DEVICE" \
  --weights weights/baseline.pt,weights/focal.pt,weights/cbam.pt,weights/dams.pt,weights/cf.pt

# 阶段7: FPS 基准 + 保存 CF-YOLOv8 可视化结果图 (--benchmark 不落盘, 需分开调用)
python inf.py --dataset "$DATASET" --weights weights/cf.pt --device "$DEVICE" --benchmark
python inf.py --dataset "$DATASET" --weights weights/cf.pt --device "$DEVICE" --save
