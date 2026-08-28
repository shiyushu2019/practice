#!/usr/bin/env bash
# 全量训练 + 验证 + 推理 (数据集: 默认混合 BCCD + TXL-PBC; 默认 GPU 0 / batch 32 / 100 轮)
# 各阶段需依次执行 (存在顺序依赖), 不要并行。

# 阶段1: 基线对照 YOLOv8n
python train.py --variant baseline --epochs 100 --dataset mixed --device 0

# 阶段2: YOLOv8n + CBAM
python train.py --variant cbam --epochs 100 --dataset mixed --device 0

# 阶段3: CF-YOLOv8 (CBAM+Focal), 与 baseline/cbam 一致单阶段训练 100 轮 (freeze=0)
python train.py --variant cf --epochs 100 --dataset mixed --device 0

# 阶段4: 在混合测试集上验证所有变体
python eval.py --split test --dataset mixed --device 0 \
  --weights weights/baseline.pt,weights/cbam.pt,weights/cf.pt

# 阶段5: 在混合测试集上推理 (CF-YOLOv8)
python inf.py --dataset mixed --weights weights/cf.pt --device 0
