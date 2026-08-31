#!/usr/bin/env bash
# 全量训练 + 验证 + 推理 (数据集: 默认混合 BCCD + TXL-PBC; 默认 GPU 1 / batch 16 / 100 轮)
# 消融四组: baseline / +Focal Loss / +CBAM / CF-YOLOv8; 各阶段需依次执行, 不要并行。

# 阶段1: 基线对照 YOLOv8n
python train.py --variant baseline --epochs 100 --dataset mixed --device 1

# 阶段2: YOLOv8n + Focal Loss (消融)
python train.py --variant focal --epochs 100 --dataset mixed --device 1

# 阶段3: YOLOv8n + CBAM
python train.py --variant cbam --epochs 100 --dataset mixed --device 1

# 阶段4: CF-YOLOv8 (CBAM+Focal), 按文档配置 (Adam/lr0=0.001/batch16/patience10/先冻结后解冻)
python train.py --variant cf --epochs 100 --dataset mixed --device 1 \
  --gamma 2.0 --alpha 0.25 --batch 16 --freeze-epochs 10

# 阶段5: 在混合测试集上验证所有变体
python eval.py --split test --dataset mixed --device 1 \
  --weights weights/baseline.pt,weights/focal.pt,weights/cbam.pt,weights/cf.pt

# 阶段6: 在混合测试集上推理 (CF-YOLOv8)
python inf.py --dataset mixed --weights weights/cf.pt --device 1
